import asyncio

from apps.app1.views import example_view

# import anything from the project

async def main():
    print("here.....")
    await example_view()