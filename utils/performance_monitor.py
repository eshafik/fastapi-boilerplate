import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor API performance and generate alerts"""

    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts
        self.thresholds = {
            'response_time_ms': 5000,  # 5 seconds
            'error_rate_percent': 5,  # 5% error rate
            'memory_usage_mb': 1000,  # 1GB memory
            'queue_length': 50  # Request queue length
        }

    def record_request(self, endpoint: str, duration_ms: int, success: bool, user_id: str = None):
        """Record request metrics"""

        timestamp = time.time()
        metric = {
            'timestamp': timestamp,
            'endpoint': endpoint,
            'duration_ms': duration_ms,
            'success': success,
            'user_id': user_id
        }

        self.metrics['requests'].append(metric)

        # Check for performance issues
        self._check_thresholds()

        # Clean old metrics (keep last hour)
        cutoff = timestamp - 3600
        self.metrics['requests'] = [m for m in self.metrics['requests'] if m['timestamp'] > cutoff]

    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the last N minutes"""

        cutoff = time.time() - (minutes * 60)
        recent_requests = [m for m in self.metrics['requests'] if m['timestamp'] > cutoff]

        if not recent_requests:
            return {
                "period_minutes": minutes,
                "total_requests": 0,
                "avg_response_time_ms": 0,
                "success_rate_percent": 100,
                "requests_per_minute": 0
            }

        # Calculate metrics
        total_requests = len(recent_requests)
        successful_requests = sum(1 for r in recent_requests if r['success'])
        total_duration = sum(r['duration_ms'] for r in recent_requests)

        avg_response_time = total_duration / total_requests
        success_rate = (successful_requests / total_requests) * 100
        requests_per_minute = total_requests / minutes

        # Endpoint breakdown
        endpoint_stats = defaultdict(list)
        for req in recent_requests:
            endpoint_stats[req['endpoint']].append(req)

        endpoint_summary = {}
        for endpoint, reqs in endpoint_stats.items():
            endpoint_summary[endpoint] = {
                "requests": len(reqs),
                "avg_duration_ms": sum(r['duration_ms'] for r in reqs) / len(reqs),
                "success_rate": (sum(1 for r in reqs if r['success']) / len(reqs)) * 100
            }

        return {
            "period_minutes": minutes,
            "total_requests": total_requests,
            "avg_response_time_ms": round(avg_response_time, 2),
            "success_rate_percent": round(success_rate, 2),
            "requests_per_minute": round(requests_per_minute, 2),
            "endpoint_breakdown": endpoint_summary,
            "recent_alerts": list(self.alerts)[-5:]  # Last 5 alerts
        }

    def _check_thresholds(self):
        """Check if any thresholds are exceeded"""

        # Get recent requests (last 5 minutes)
        cutoff = time.time() - 300
        recent_requests = [m for m in self.metrics['requests'] if m['timestamp'] > cutoff]

        if len(recent_requests) < 5:  # Need minimum data
            return

        # Check average response time
        avg_response_time = sum(r['duration_ms'] for r in recent_requests) / len(recent_requests)
        if avg_response_time > self.thresholds['response_time_ms']:
            self._create_alert(
                'high_response_time',
                f'Average response time: {avg_response_time:.0f}ms (threshold: {self.thresholds["response_time_ms"]}ms)'
            )

        # Check error rate
        error_count = sum(1 for r in recent_requests if not r['success'])
        error_rate = (error_count / len(recent_requests)) * 100
        if error_rate > self.thresholds['error_rate_percent']:
            self._create_alert(
                'high_error_rate',
                f'Error rate: {error_rate:.1f}% (threshold: {self.thresholds["error_rate_percent"]}%)'
            )

    def _create_alert(self, alert_type: str, message: str):
        """Create a performance alert"""

        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': 'warning'
        }

        self.alerts.append(alert)
        logger.warning(f"Performance Alert [{alert_type}]: {message}")

performance_monitor = PerformanceMonitor()