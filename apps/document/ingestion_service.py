import re
from urllib.parse import urlparse
import tempfile
import os

import aiohttp
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PDFReader
from llama_index.embeddings.openai import OpenAIEmbedding
import openai
from typing import List, Dict, Any, Tuple, Optional
import uuid
from datetime import datetime
import tiktoken
import logging
import asyncio
import aiofiles

from apps.document.models import Document, DocumentChunk, DocumentType
from config import settings
from config.settings import OPENAI_EMBEDDING_MODEL, OPENAI_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self):
        self.embedding_model = OpenAIEmbedding(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            paragraph_separator="\n\n",
            secondary_chunking_regex="[.!?]+",
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))

    async def extract_web_content(self, url: str, max_pages: Optional[int] = 1) -> List[LlamaDocument]:
        """Extract content from web URLs with intelligent crawling"""
        try:
            # Use max_pages or default limit
            page_limit = min(max_pages or settings.MAX_WEB_PAGES, settings.MAX_WEB_PAGES)

            # Try TrafilaturaWebReader first (best for production)
            try:
                print('//'*100)
                documents = await self._extract_with_trafilatura(url, page_limit)
                if documents:
                    logger.info(f"Trafilatura extracted {len(documents)} pages from {url}")
                    return documents
            except Exception as e:
                logger.warning(f"Trafilatura extraction failed: {str(e)}")

            # Fallback to custom intelligent crawler
            documents = await self._extract_with_intelligent_crawler(url, page_limit)
            logger.info(f"Intelligent crawler extracted {len(documents)} pages from {url}")
            return documents

        except Exception as e:
            logger.error(f"Failed to extract web content from {url}: {str(e)}")
            raise

    async def _extract_with_trafilatura(self, start_url: str, max_pages: int) -> List[LlamaDocument]:
        """Extract web content using TrafilaturaWebReader (production-grade)"""
        try:
            from llama_index.readers.web import TrafilaturaWebReader

            # Get related URLs for crawling
            urls_to_crawl = await self._discover_related_urls(start_url, max_pages)
            print('^^'*100, 'urls to crawl', urls_to_crawl)

            reader = TrafilaturaWebReader()
            documents = reader.load_data(urls_to_crawl)

            # Filter out empty documents and add metadata
            filtered_docs = []
            for i, doc in enumerate(documents):
                if doc.text and len(doc.text.strip()) > 100:  # Minimum content threshold
                    doc.metadata.update({
                        "source_url": urls_to_crawl[i] if i < len(urls_to_crawl) else start_url,
                        "extraction_method": "trafilatura",
                        "page_index": i,
                        "crawl_depth": self._calculate_depth(start_url,
                                                             urls_to_crawl[i] if i < len(urls_to_crawl) else start_url)
                    })
                    filtered_docs.append(doc)

            return filtered_docs

        except ImportError:
            logger.warning(
                "TrafilaturaWebReader not available. Install with: pip install llama-index-readers-web[trafilatura]")
            raise

    async def _extract_with_intelligent_crawler(self, start_url: str, max_pages: int) -> List[LlamaDocument]:
        """Custom intelligent web crawler with smart link discovery"""
        import asyncio
        import aiohttp
        import re
        from urllib.parse import urljoin, urlparse, parse_qs
        from bs4 import BeautifulSoup

        documents = []
        visited_urls = set()
        urls_to_visit = [start_url]
        base_domain = urlparse(start_url).netloc

        # Configure session with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

        timeout = aiohttp.ClientTimeout(total=30, connect=10)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            while urls_to_visit and len(documents) < max_pages:
                current_url = urls_to_visit.pop(0)

                # Skip if already visited or invalid
                if current_url in visited_urls or not self._is_valid_url(current_url, base_domain):
                    continue

                visited_urls.add(current_url)

                try:
                    # Extract content from current page
                    content, new_urls = await self._extract_page_content(session, current_url, base_domain)

                    if content and len(content.strip()) > 100:
                        documents.append(LlamaDocument(
                            text=content,
                            metadata={
                                "source_url": current_url,
                                "extraction_method": "intelligent_crawler",
                                "page_index": len(documents),
                                "crawl_depth": self._calculate_depth(start_url, current_url),
                                "title": await self._extract_page_title(session, current_url)
                            }
                        ))

                    # Add new URLs to crawl queue (prioritized)
                    prioritized_urls = self._prioritize_urls(new_urls, start_url, current_url)
                    for new_url in prioritized_urls:
                        if new_url not in visited_urls and new_url not in urls_to_visit:
                            urls_to_visit.append(new_url)

                    # Respect rate limiting
                    await asyncio.sleep(1)  # 1 second delay between requests

                except Exception as e:
                    logger.warning(f"Failed to extract from {current_url}: {str(e)}")
                    continue

        return documents

    async def _discover_related_urls(self, start_url: str, max_urls: int) -> List[str]:
        """Discover related URLs for crawling using sitemap and intelligent discovery"""
        urls = [start_url]
        base_domain = urlparse(start_url).netloc

        # Try to get sitemap URLs first
        sitemap_urls = await self._get_sitemap_urls(start_url, max_urls - 1)
        urls.extend(sitemap_urls)

        # If not enough URLs from sitemap, discover more
        if len(urls) < max_urls:
            discovered_urls = await self._discover_urls_from_page(start_url, base_domain, max_urls - len(urls))
            urls.extend(discovered_urls)

        return urls[:max_urls]

    async def _get_sitemap_urls(self, start_url: str, max_urls: int) -> List[str]:
        """Extract URLs from website sitemap"""
        import aiohttp
        import xml.etree.ElementTree as ET
        from urllib.parse import urljoin, urlparse

        base_url = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')  # Check robots.txt for sitemap
        ]

        urls = []

        async with aiohttp.ClientSession() as session:
            for sitemap_url in sitemap_urls:
                try:
                    async with session.get(sitemap_url, timeout=10) as response:
                        if response.status == 200:
                            content = await response.text()

                            if sitemap_url.endswith('robots.txt'):
                                # Extract sitemap URLs from robots.txt
                                for line in content.split('\n'):
                                    if line.lower().startswith('sitemap:'):
                                        sitemap_url = line.split(':', 1)[1].strip()
                                        urls.extend(await self._parse_sitemap(session, sitemap_url, max_urls))
                            else:
                                # Parse XML sitemap
                                urls.extend(await self._parse_sitemap_xml(content, max_urls))

                            if len(urls) >= max_urls:
                                break

                except Exception as e:
                    logger.debug(f"Failed to fetch sitemap {sitemap_url}: {str(e)}")
                    continue

        return urls[:max_urls]

    async def _parse_sitemap_xml(self, xml_content: str, max_urls: int) -> List[str]:
        """Parse XML sitemap content"""
        import xml.etree.ElementTree as ET

        urls = []
        try:
            root = ET.fromstring(xml_content)

            # Handle different sitemap namespaces
            for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
                loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text)
                    if len(urls) >= max_urls:
                        break

            # Also check for sitemap index
            for sitemap_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
                loc_elem = sitemap_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                if loc_elem is not None and loc_elem.text:
                    # Recursively parse sub-sitemaps
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        sub_urls = await self._parse_sitemap(session, loc_elem.text, max_urls - len(urls))
                        urls.extend(sub_urls)
                        if len(urls) >= max_urls:
                            break

        except ET.ParseError as e:
            logger.debug(f"Failed to parse sitemap XML: {str(e)}")

        return urls[:max_urls]

    async def _parse_sitemap(self, session: aiohttp.ClientSession, sitemap_url: str, max_urls: int) -> List[str]:
        """Parse a single sitemap URL"""
        try:
            async with session.get(sitemap_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    return await self._parse_sitemap_xml(content, max_urls)
        except Exception as e:
            logger.debug(f"Failed to parse sitemap {sitemap_url}: {str(e)}")

        return []

    async def _discover_urls_from_page(self, start_url: str, base_domain: str, max_urls: int) -> List[str]:
        """Discover URLs by crawling from the start page"""
        import aiohttp
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        urls = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(start_url, timeout=15) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')

                        # Find all links
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            full_url = urljoin(start_url, href)

                            if self._is_valid_url(full_url, base_domain) and full_url not in urls:
                                urls.append(full_url)
                                if len(urls) >= max_urls:
                                    break

        except Exception as e:
            logger.warning(f"Failed to discover URLs from {start_url}: {str(e)}")

        return urls

    async def _extract_page_content(self, session: aiohttp.ClientSession, url: str, base_domain: str) -> tuple[
        str, List[str]]:
        """Extract content and links from a single page"""
        from bs4 import BeautifulSoup
        from urllib.parse import urljoin

        try:
            async with session.get(url, timeout=15) as response:
                if response.status != 200:
                    return "", []

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()

                # Extract main content (try multiple selectors)
                content_selectors = [
                    'main', 'article', '.content', '#content',
                    '.post-content', '.entry-content', '.article-content'
                ]

                content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = ' '.join([elem.get_text(strip=True) for elem in elements])
                        break

                # Fallback to body content
                if not content or len(content.strip()) < 100:
                    content = soup.get_text(strip=True)

                # Clean up content
                content = re.sub(r'\s+', ' ', content)
                content = re.sub(r'\n+', '\n', content)

                # Extract new URLs
                new_urls = []
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    if self._is_valid_url(full_url, base_domain):
                        new_urls.append(full_url)

                return content, new_urls

        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {str(e)}")
            return "", []

    async def _extract_page_title(self, session: aiohttp.ClientSession, url: str) -> str:
        """Extract page title"""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    title = soup.find('title')
                    return title.get_text(strip=True) if title else ""
        except:
            pass
        return ""

    def _is_valid_url(self, url: str, base_domain: str) -> bool:
        """Check if URL is valid for crawling"""
        try:
            parsed = urlparse(url)

            # Must be same domain
            if parsed.netloc != base_domain:
                return False

            # Must be HTTP/HTTPS
            if parsed.scheme not in ['http', 'https']:
                return False

            # Skip unwanted file types
            skip_extensions = {
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.rar', '.tar', '.gz', '.mp3', '.mp4', '.avi',
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.css', '.js'
            }

            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in skip_extensions):
                return False

            # Skip unwanted paths
            skip_patterns = [
                '/admin', '/login', '/register', '/cart', '/checkout',
                '/api/', '/wp-admin', '/wp-content', '/feed',
                '?print=', '?share=', '#'
            ]

            full_url = url.lower()
            if any(pattern in full_url for pattern in skip_patterns):
                return False

            return True

        except Exception:
            return False

    def _prioritize_urls(self, urls: List[str], start_url: str, current_url: str) -> List[str]:
        """Prioritize URLs for crawling based on relevance"""

        def url_score(url):
            score = 0
            parsed = urlparse(url)
            path = parsed.path.lower()

            # Prefer pages that seem content-rich
            content_indicators = [
                'article', 'post', 'blog', 'news', 'story', 'guide',
                'tutorial', 'documentation', 'docs', 'help', 'about'
            ]

            for indicator in content_indicators:
                if indicator in path:
                    score += 10

            # Prefer shorter paths (closer to root)
            score -= path.count('/') * 2

            # Prefer pages from same directory as current
            current_dir = '/'.join(urlparse(current_url).path.split('/')[:-1])
            if path.startswith(current_dir):
                score += 5

            return score

        # Sort URLs by score (highest first)
        return sorted(urls, key=url_score, reverse=True)

    def _calculate_depth(self, start_url: str, current_url: str) -> int:
        """Calculate crawl depth from start URL"""
        start_path = urlparse(start_url).path.rstrip('/').split('/')
        current_path = urlparse(current_url).path.rstrip('/').split('/')

        return max(0, len(current_path) - len(start_path))

    async def extract_pdf_content(self, url: str, max_pages: Optional[int] = None) -> List[LlamaDocument]:
        """Extract content from PDF using multiple fallback methods with enhanced robustness"""
        temp_filename = None
        try:
            # Download PDF with enhanced error handling
            temp_filename = await self._download_pdf_async(url)
            logger.info(f"Downloaded PDF to: {temp_filename}")

            # Try multiple PDF extraction methods with enhanced fallbacks
            documents = await self._extract_pdf_with_enhanced_fallbacks(temp_filename, url, max_pages)

            if not documents:
                raise ValueError("No content could be extracted from PDF using any method")

            logger.info(f"Successfully extracted {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            logger.error(f"Failed to extract PDF content from {url}: {str(e)}")
            raise
        finally:
            # Clean up temp file
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.debug(f"Cleaned up temp file: {temp_filename}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp file {temp_filename}: {e}")

    async def _download_pdf_async(self, url: str) -> str:
        """Download PDF asynchronously with enhanced error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/octet-stream,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Upgrade-Insecure-Requests': '1'
        }

        timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 minute total, 30s connect

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            async with session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download PDF: HTTP {response.status}")

                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'octet-stream' not in content_type:
                    logger.warning(f"Unexpected content type: {content_type}")

                # Create temp file
                temp_fd, temp_filename = tempfile.mkstemp(suffix='.pdf', prefix='pdf_extract_')

                try:
                    # Write content to temp file
                    async with aiofiles.open(temp_filename, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

                    # Close the file descriptor
                    os.close(temp_fd)

                    # Verify file size
                    file_size = os.path.getsize(temp_filename)
                    if file_size == 0:
                        raise ValueError("Downloaded PDF file is empty")

                    logger.info(f"Downloaded PDF: {file_size} bytes")
                    return temp_filename

                except Exception as e:
                    # Clean up on error
                    try:
                        os.close(temp_fd)
                    except:
                        pass
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    raise

    async def _extract_pdf_with_enhanced_fallbacks(self, filepath: str, url: str, max_pages: Optional[int] = None) -> \
    List[LlamaDocument]:
        """Try multiple PDF extraction methods with enhanced fallbacks and error handling"""
        documents = []
        extraction_methods = []

        # Method 1: PyMuPDF (most robust, try first)
        try:
            documents = await self._extract_with_pymupdf(filepath, max_pages)
            if documents:
                extraction_methods.append("pymupdf")
                logger.info(f"PyMuPDF extracted {len(documents)} pages")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")

        # Method 2: pdfplumber (good for structured content)
        if not documents:
            try:
                documents = await self._extract_with_pdfplumber(filepath, max_pages)
                if documents:
                    extraction_methods.append("pdfplumber")
                    logger.info(f"pdfplumber extracted {len(documents)} pages")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {str(e)}")

        # Method 3: LlamaIndex PDFReader
        if not documents:
            try:
                documents = await self._extract_with_llamaindex(filepath, max_pages)
                if documents:
                    extraction_methods.append("llamaindex")
                    logger.info(f"LlamaIndex PDFReader extracted {len(documents)} pages")
            except Exception as e:
                logger.warning(f"LlamaIndex PDFReader failed: {str(e)}")

        # Method 4: PyPDF2/PyPDF4 (basic but sometimes works)
        if not documents:
            try:
                documents = await self._extract_with_pypdf(filepath, max_pages)
                if documents:
                    extraction_methods.append("pypdf")
                    logger.info(f"PyPDF extracted {len(documents)} pages")
            except Exception as e:
                logger.warning(f"PyPDF extraction failed: {str(e)}")

        # Method 5: pdfminer3k (alternative parser)
        if not documents:
            try:
                documents = await self._extract_with_pdfminer(filepath, max_pages)
                if documents:
                    extraction_methods.append("pdfminer")
                    logger.info(f"pdfminer extracted {len(documents)} pages")
            except Exception as e:
                logger.warning(f"pdfminer extraction failed: {str(e)}")

        # Method 6: OCR fallback for scanned/image PDFs
        if not documents:
            try:
                documents = await self._extract_pdf_with_enhanced_ocr(filepath, max_pages)
                if documents:
                    extraction_methods.append("ocr")
                    logger.info(f"OCR extracted {len(documents)} pages")
            except Exception as e:
                logger.warning(f"OCR extraction failed: {str(e)}")

        # Add metadata to all documents
        for doc in documents:
            doc.metadata.update({
                "source_url": url,
                "extraction_methods": extraction_methods,
                "file_path": filepath
            })

        # Apply page limit if specified
        if max_pages and len(documents) > max_pages:
            documents = documents[:max_pages]
            logger.info(f"Limited PDF to {max_pages} pages")

        return documents

    async def _extract_with_pymupdf(self, filepath: str, max_pages: Optional[int] = None) -> List[LlamaDocument]:
        """Extract using PyMuPDF with enhanced error handling"""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF not available, install with: pip install PyMuPDF")
            return []

        documents = []
        doc = None

        try:
            # Try opening with different modes
            try:
                doc = fitz.open(filepath)
            except Exception:
                # Try with different flags for corrupted PDFs
                doc = fitz.open(filepath, filetype="pdf")

            if doc.is_encrypted:
                logger.warning("PDF is encrypted, attempting to decrypt")
                if not doc.authenticate(""):  # Try empty password
                    logger.error("Could not decrypt PDF")
                    return []

            page_count = min(len(doc), max_pages or 1000)

            for page_num in range(page_count):
                try:
                    page = doc[page_num]

                    # Try multiple text extraction methods
                    text = ""

                    # Method 1: Standard text extraction
                    try:
                        text = page.get_text()
                    except:
                        pass

                    # Method 2: Text with layout preservation
                    if not text or len(text.strip()) < 50:
                        try:
                            text = page.get_text("text", sort=True)
                        except:
                            pass

                    # Method 3: Text blocks
                    if not text or len(text.strip()) < 50:
                        try:
                            blocks = page.get_text("dict")["blocks"]
                            text_parts = []
                            for block in blocks:
                                if "lines" in block:
                                    for line in block["lines"]:
                                        for span in line["spans"]:
                                            text_parts.append(span["text"])
                            text = " ".join(text_parts)
                        except:
                            pass

                    if text and text.strip():
                        # Clean up text
                        text = re.sub(r'\s+', ' ', text.strip())

                        documents.append(LlamaDocument(
                            text=text,
                            metadata={
                                "page_label": str(page_num + 1),
                                "source": filepath,
                                "extraction_method": "pymupdf",
                                "page_index": page_num
                            }
                        ))

                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {str(e)}")
            return []
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass

        return documents

    async def _extract_with_pdfplumber(self, filepath: str, max_pages: Optional[int] = None) -> List[LlamaDocument]:
        """Extract using pdfplumber with enhanced error handling"""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not available, install with: pip install pdfplumber")
            return []

        documents = []

        try:
            with pdfplumber.open(filepath, strict=False) as pdf:
                page_count = min(len(pdf.pages), max_pages or 1000)

                for page_num in range(page_count):
                    try:
                        page = pdf.pages[page_num]

                        # Try multiple extraction methods
                        text = ""

                        # Method 1: Standard extraction
                        try:
                            text = page.extract_text()
                        except:
                            pass

                        # Method 2: Extract with layout
                        if not text or len(text.strip()) < 50:
                            try:
                                text = page.extract_text(layout=True)
                            except:
                                pass

                        # Method 3: Extract words and join
                        if not text or len(text.strip()) < 50:
                            try:
                                words = page.extract_words()
                                text = " ".join([word["text"] for word in words])
                            except:
                                pass

                        if text and text.strip():
                            # Clean up text
                            text = re.sub(r'\s+', ' ', text.strip())

                            documents.append(LlamaDocument(
                                text=text,
                                metadata={
                                    "page_label": str(page_num + 1),
                                    "source": filepath,
                                    "extraction_method": "pdfplumber",
                                    "page_index": page_num
                                }
                            ))

                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num} with pdfplumber: {str(e)}")
                        continue

        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {str(e)}")
            return []

        return documents

    async def _extract_with_llamaindex(self, filepath: str, max_pages: Optional[int] = None) -> List[LlamaDocument]:
        """Extract using LlamaIndex PDFReader with enhanced error handling"""
        try:
            from llama_index.readers.file import PDFReader
        except ImportError:
            logger.warning("LlamaIndex PDFReader not available")
            return []

        documents = []

        try:
            reader = PDFReader()
            documents = reader.load_data(filepath)

            # Apply page limit
            if max_pages and len(documents) > max_pages:
                documents = documents[:max_pages]

            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "page_label": str(i + 1),
                    "source": filepath,
                    "extraction_method": "llamaindex",
                    "page_index": i
                })

        except Exception as e:
            logger.error(f"LlamaIndex PDFReader extraction failed: {str(e)}")
            return []

        return documents

    async def _extract_with_pypdf(self, filepath: str, max_pages: Optional[int] = None) -> List[LlamaDocument]:
        """Extract using PyPDF2/PyPDF4 with enhanced error handling"""
        documents = []

        # Try PyPDF4 first, then PyPDF2
        pdf_libraries = []

        try:
            from PyPDF4 import PdfFileReader
            pdf_libraries.append(("PyPDF4", PdfFileReader))
        except ImportError:
            pass

        try:
            from PyPDF2 import PdfReader as PyPDF2Reader
            pdf_libraries.append(("PyPDF2", PyPDF2Reader))
        except ImportError:
            pass

        if not pdf_libraries:
            logger.warning("Neither PyPDF4 nor PyPDF2 available")
            return []

        for lib_name, PdfReaderClass in pdf_libraries:
            try:
                with open(filepath, 'rb') as file:
                    if lib_name == "PyPDF4":
                        reader = PdfReaderClass(file, strict=False)
                        pages = reader.pages
                    else:  # PyPDF2
                        reader = PdfReaderClass(file, strict=False)
                        pages = reader.pages

                    page_count = min(len(pages), max_pages or 1000)

                    for page_num in range(page_count):
                        try:
                            page = pages[page_num]
                            text = page.extract_text()

                            if text and text.strip():
                                # Clean up text
                                text = re.sub(r'\s+', ' ', text.strip())

                                documents.append(LlamaDocument(
                                    text=text,
                                    metadata={
                                        "page_label": str(page_num + 1),
                                        "source": filepath,
                                        "extraction_method": lib_name.lower(),
                                        "page_index": page_num
                                    }
                                ))

                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num} with {lib_name}: {str(e)}")
                            continue

                # If we got documents, break out of the library loop
                if documents:
                    logger.info(f"Successfully extracted with {lib_name}")
                    break

            except Exception as e:
                logger.warning(f"{lib_name} extraction failed: {str(e)}")
                continue

        return documents

    async def _extract_with_pdfminer(self, filepath: str, max_pages: Optional[int] = None) -> List[LlamaDocument]:
        """Extract using pdfminer3k with enhanced error handling"""
        try:
            from pdfminer.high_level import extract_text_to_fp, extract_pages
            from pdfminer.layout import LTTextContainer
            from io import StringIO
        except ImportError:
            logger.warning("pdfminer3k not available, install with: pip install pdfminer3k")
            return []

        documents = []

        try:
            # Method 1: Extract all text at once
            try:
                with open(filepath, 'rb') as file:
                    output_string = StringIO()
                    extract_text_to_fp(file, output_string, maxpages=max_pages)
                    text = output_string.getvalue()

                    if text and text.strip():
                        # Split into pages (rough approximation)
                        pages = text.split('\f')  # Form feed character often separates pages

                        for i, page_text in enumerate(pages):
                            if page_text.strip():
                                page_text = re.sub(r'\s+', ' ', page_text.strip())
                                documents.append(LlamaDocument(
                                    text=page_text,
                                    metadata={
                                        "page_label": str(i + 1),
                                        "source": filepath,
                                        "extraction_method": "pdfminer",
                                        "page_index": i
                                    }
                                ))
            except Exception as e:
                logger.warning(f"pdfminer bulk extraction failed: {str(e)}")

            # Method 2: Page-by-page extraction
            if not documents:
                try:
                    with open(filepath, 'rb') as file:
                        page_num = 0
                        for page_layout in extract_pages(file, maxpages=max_pages):
                            text_parts = []
                            for element in page_layout:
                                if isinstance(element, LTTextContainer):
                                    text_parts.append(element.get_text())

                            page_text = ' '.join(text_parts)
                            if page_text.strip():
                                page_text = re.sub(r'\s+', ' ', page_text.strip())
                                documents.append(LlamaDocument(
                                    text=page_text,
                                    metadata={
                                        "page_label": str(page_num + 1),
                                        "source": filepath,
                                        "extraction_method": "pdfminer_pages",
                                        "page_index": page_num
                                    }
                                ))
                            page_num += 1
                except Exception as e:
                    logger.warning(f"pdfminer page extraction failed: {str(e)}")

        except Exception as e:
            logger.error(f"pdfminer extraction failed: {str(e)}")
            return []

        return documents

    async def _extract_pdf_with_enhanced_ocr(self, filepath: str, max_pages: Optional[int] = None) -> List[
        LlamaDocument]:
        """Extract PDF content using OCR as last resort with enhanced processing"""
        try:
            import fitz  # PyMuPDF for image conversion
            from PIL import Image, ImageEnhance, ImageFilter
            import pytesseract
            import io
        except ImportError:
            logger.warning("OCR dependencies not available. Install with: pip install PyMuPDF pillow pytesseract")
            return []

        documents = []
        doc = None

        try:
            doc = fitz.open(filepath)
            page_count = min(len(doc), max_pages or 1000)

            for page_num in range(page_count):
                try:
                    page = doc[page_num]

                    # Convert page to high-resolution image
                    # Higher DPI for better OCR results
                    matrix = fitz.Matrix(3, 3)  # 3x scale for better OCR
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    # Image preprocessing for better OCR
                    # Convert to grayscale
                    if image.mode != 'L':
                        image = image.convert('L')

                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(2.0)

                    # Sharpen image
                    image = image.filter(ImageFilter.SHARPEN)

                    # Resize if too small
                    width, height = image.size
                    if width < 1000 or height < 1000:
                        scale_factor = max(1000 / width, 1000 / height)
                        new_size = (int(width * scale_factor), int(height * scale_factor))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)

                    # Extract text using OCR with optimized settings
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()_+-={}[]|\\:";\'<>?,./ '

                    try:
                        text = pytesseract.image_to_string(image, config=custom_config)
                    except:
                        # Fallback with default config
                        text = pytesseract.image_to_string(image)

                    if text and text.strip():
                        # Clean up OCR text
                        text = re.sub(r'\s+', ' ', text.strip())
                        # Remove common OCR artifacts
                        text = re.sub(r'[^\w\s\.,!?;:\'"()\-\[\]{}]', '', text)

                        if len(text.strip()) > 20:  # Minimum threshold for OCR content
                            documents.append(LlamaDocument(
                                text=text,
                                metadata={
                                    "page_label": str(page_num + 1),
                                    "source": filepath,
                                    "extraction_method": "enhanced_ocr",
                                    "page_index": page_num
                                }
                            ))

                except Exception as e:
                    logger.warning(f"OCR failed for page {page_num}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed: {str(e)}")
            return []
        finally:
            if doc:
                try:
                    doc.close()
                except:
                    pass

        return documents

    async def process_text_content(self, content: str, title: str) -> List[LlamaDocument]:
        """Process raw text content"""
        document = LlamaDocument(
            text=content,
            metadata={"title": title}
        )
        return [document]

    async def chunk_documents(self, documents: List[LlamaDocument]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        all_chunks = []

        for doc_idx, document in enumerate(documents):
            # Split document into nodes/chunks
            nodes = self.text_splitter.get_nodes_from_documents([document])

            for chunk_idx, node in enumerate(nodes):
                chunk_data = {
                    "content": node.text,
                    "token_count": self.count_tokens(node.text),
                    "chunk_index": chunk_idx,
                    "metadata": {
                        **node.metadata,
                        "document_index": doc_idx,
                        "chunk_index": chunk_idx
                    }
                }
                all_chunks.append(chunk_data)

        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        try:
            embeddings = await self.embedding_model.aget_text_embedding_batch(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    async def prepare_chunks_for_indexing(
            self,
            chunks: List[Dict[str, Any]],
            document: Document,
            embeddings: List[List[float]]
    ) -> Tuple[List[DocumentChunk], List[Dict[str, Any]]]:
        """Prepare chunks for database and Elasticsearch storage"""
        db_chunks = []
        es_chunks = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Generate unique ES document ID
            es_doc_id = f"{document.id}_{i}"

            # Database chunk
            db_chunk = DocumentChunk(
                document=document,
                chunk_index=i,
                content=chunk["content"],
                token_count=chunk["token_count"],
                page_number=chunk["metadata"].get("page_label"),
                section_title=chunk["metadata"].get("section_title"),
                url=document.source_url if document.document_type == DocumentType.WEB else None,
                es_doc_id=es_doc_id
            )
            db_chunks.append(db_chunk)

            # Elasticsearch chunk
            es_chunk = {
                "chunk_id": es_doc_id,
                "document_id": str(document.id),
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": {
                    "document_title": document.title,
                    "document_type": document.document_type.value,
                    "page_number": chunk["metadata"].get("page_label"),
                    "section_title": chunk["metadata"].get("section_title"),
                    "url": document.source_url,
                    "chunk_index": i,
                    "token_count": chunk["token_count"]
                },
                "created_at": datetime.utcnow().isoformat()
            }
            es_chunks.append(es_chunk)

        return db_chunks, es_chunks


# Global service instance
ingestion_service = IngestionService()