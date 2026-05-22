import os
import json
import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Tuple
from urllib.parse import quote_plus, urlparse
from datetime import datetime
from PIL import Image
from io import BytesIO
import backend.job_store as job_store

# Set up logging
logger = logging.getLogger("scraper")

# Global variables for per-domain rate limiting
domain_last_request_time = {}
domain_locks = {}
domain_locks_lock = asyncio.Lock()

async def rate_limit_domain(url: str):
    domain = urlparse(url).netloc
    if not domain:
        return
        
    async with domain_locks_lock:
        if domain not in domain_locks:
            domain_locks[domain] = asyncio.Lock()
        lock = domain_locks[domain]
        
    async with lock:
        last_time = domain_last_request_time.get(domain, 0.0)
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - last_time
        if elapsed < 0.2:
            await asyncio.sleep(0.2 - elapsed)
        domain_last_request_time[domain] = asyncio.get_event_loop().time()

def simplify_query(query: str) -> str:
    # Remove common prefix noise
    q = query.lower()
    prefixes = ["images of ", "photos of ", "pictures of ", "image of ", "photo of ", "picture of ", "dataset of ", "datasets of "]
    for prefix in prefixes:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
            
    # Remove common preposition phrases
    prepositions = [" in ", " on ", " with ", " at ", " by ", " near "]
    for prep in prepositions:
        if prep in q:
            q = q.split(prep)[0]
            
    # Keep first 3 words if still long
    words = q.split()
    if len(words) > 4:
        return " ".join(words[:3])
    return q.strip()

async def download_image(
    session: aiohttp.ClientSession,
    url: str,
    output_path: str,
    semaphore: asyncio.Semaphore,
    timeout: int = 5,
    retries: int = 3
) -> bool:
    """Downloads a single image with timeout, concurrency limits, and retries."""
    async with semaphore:
        for attempt in range(retries):
            try:
                # Per-domain rate limit (200ms delay)
                await rate_limit_domain(url)
                
                # Add a realistic user agent
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to download {url}: HTTP {response.status} (Attempt {attempt+1}/{retries})")
                        continue
                        
                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        logger.warning(f"Skip non-image content {url}: {content_type}")
                        return False
                        
                    data = await response.read()
                    
                    # Verify it's a valid image using Pillow
                    try:
                        img = Image.open(BytesIO(data))
                        img.verify()  # Verify image integrity
                        
                        # Re-open because verify() closes/invalidates the image object
                        img = Image.open(BytesIO(data))
                        
                        # Convert PNG/RGBA/etc. to RGB and save as JPG
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        
                        # Atomic write: write to temp and rename
                        temp_path = output_path + ".tmp"
                        img.save(temp_path, "JPEG")
                        os.replace(temp_path, output_path)
                        return True
                    except Exception as e:
                        logger.warning(f"Corrupted image downloaded from {url}: {e}")
                        continue
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading {url} (Attempt {attempt+1}/{retries})")
            except Exception as e:
                logger.warning(f"Error downloading {url} (Attempt {attempt+1}/{retries}): {e}")
                
            # Backoff before retry
            await asyncio.sleep(0.5)
            
        return False

def _scrape_google_bing_playwright_sync(query: str, count: int) -> List[Dict[str, str]]:
    """Sync Playwright scraper that runs in its own thread (avoids Windows event loop issues)."""
    import time
    from playwright.sync_api import sync_playwright

    logger.info("Initializing Playwright scraper (sync)...")
    urls = []
    seen_urls = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()

            # Scrape Bing Images (very easy to extract direct high-res URLs from JSON attribute 'm')
            search_url = f"https://www.bing.com/images/search?q={quote_plus(query)}"
            logger.info(f"Navigating to: {search_url}")
            page.goto(search_url, wait_until="domcontentloaded")

            # Scroll down to load enough images (scaled proportional to target count)
            scroll_count = max(5, count // 20)
            for _ in range(scroll_count):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1.0)

            # Extract links
            elements = page.locator("a.iusc").all()
            logger.info(f"Found {len(elements)} image elements on page.")

            for elem in elements:
                m_attr = elem.get_attribute("m")
                if m_attr:
                    try:
                        m_data = json.loads(m_attr)
                        murl = m_data.get("murl")
                        if murl and murl.startswith("http") and murl not in seen_urls:
                            seen_urls.add(murl)
                            urls.append({
                                "url": murl,
                                "source": "Bing Scraper (Playwright)"
                            })
                            if len(urls) >= count * 2:
                                break
                    except Exception as e:
                        logger.warning(f"Failed to parse Bing iusc attr: {e}")

            # If we don't have enough, try Google Images
            if len(urls) < count:
                logger.info("Fewer URLs than needed. Trying Google Images...")
                google_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
                page.goto(google_url, wait_until="domcontentloaded")

                for _ in range(scroll_count):
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    time.sleep(1.0)

                # Google images: target specific selectors to avoid junk/logos
                images = page.locator("img.YQ4gaf, img.rg_i, div[data-ri] img").all()
                for img in images:
                    src = img.get_attribute("src")
                    if not src:
                        src = img.get_attribute("data-src")

                    if src and src.startswith("http") and src not in seen_urls:
                        seen_urls.add(src)
                        urls.append({
                            "url": src,
                            "source": "Google Images (Playwright)"
                        })
                        if len(urls) >= count * 2:
                            break

            browser.close()
    except Exception as e:
        logger.error(f"Playwright scraping failed: {e}", exc_info=True)

    return urls[:count * 2]

async def scrape_google_bing_playwright(query: str, count: int) -> List[Dict[str, str]]:
    """Async wrapper that runs sync Playwright in a thread to avoid Windows event loop issues."""
    return await asyncio.to_thread(_scrape_google_bing_playwright_sync, query, count)

async def scrape_bing_api(query: str, count: int, api_key: str) -> List[Dict[str, str]]:
    """Scrapes images using official Bing Image Search API."""
    logger.info("Using Bing Search API...")
    url = "https://api.bing.microsoft.com/v7.0/images/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": min(count * 2, 150),  # Fetch excess to buffer against bad downloads
        "safeSearch": "Off"
    }
    
    urls = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    for item in data.get("value", []):
                        content_url = item.get("contentUrl")
                        source_page = item.get("hostPageUrl")
                        if content_url and content_url.startswith("http"):
                            urls.append({
                                "url": content_url,
                                "source": source_page or "Bing API"
                            })
                else:
                    logger.error(f"Bing API returned status code: {response.status}")
    except Exception as e:
        logger.error(f"Bing API call failed: {e}", exc_info=True)
        
    return urls

async def scrape_and_download(
    query: str,
    count: int,
    output_dir: str,
    job_id: str,
    on_progress_update=None
) -> Tuple[int, str]:
    """
    Scrapes and downloads images concurrently.
    Returns: (number of downloaded images, path to metadata.json)
    """
    raw_dir = os.path.join(output_dir, job_id, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    # Check cancellation before starting
    if job_store.is_job_cancelled(job_id):
        raise ValueError("Job was cancelled by the user.")
        
    # 1. Check API key vs Playwright fallback
    api_key = os.getenv("BING_API_KEY", "").strip()
    image_sources = []
    
    if api_key:
        try:
            image_sources = await scrape_bing_api(query, count, api_key)
        except Exception as e:
            logger.warning(f"Bing API scraping failed, falling back to Playwright: {e}")
            
    # Fallback to Playwright if no API key or API call yielded no results
    if not image_sources:
        if api_key:
            logger.info("Bing API returned 0 results. Falling back to Playwright.")
        else:
            logger.info("No Bing API key found in environment. Defaulting to Playwright.")
        image_sources = await scrape_google_bing_playwright(query, count)
        
    # Deduplicate candidate URLs across engines
    deduped_sources = []
    seen = set()
    for src in image_sources:
        u = src["url"]
        if u not in seen:
            seen.add(u)
            deduped_sources.append(src)
    image_sources = deduped_sources
        
    if not image_sources:
        logger.error("No image sources could be found from either API or Playwright Scraper.")
        return 0, ""

    # Load existing metadata if appending
    metadata = {}
    metadata_file = os.path.join(raw_dir, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Loaded {len(metadata)} existing metadata entries from {metadata_file}")
        except Exception as e:
            logger.error(f"Failed to load existing metadata.json: {e}", exc_info=True)

    initial_metadata_count = len(metadata)

    # Determine the starting index for new images
    def get_start_idx():
        existing_indices = []
        if os.path.exists(raw_dir):
            for f in os.listdir(raw_dir):
                name, ext = os.path.splitext(f)
                if ext.lower() == ".jpg" and name.isdigit():
                    existing_indices.append(int(name))
        return max(existing_indices) + 1 if existing_indices else 0

    max_concurrency = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "10"))
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_downloads(sources, start_idx_val):
        downloaded_new = 0
        active_tasks = []
        async with aiohttp.ClientSession() as session:
            for idx_offset, src_info in enumerate(sources):
                # Check cancellation in the loop
                if job_store.is_job_cancelled(job_id):
                    logger.info(f"Job {job_id} cancelled during download. Aborting.")
                    raise ValueError("Job was cancelled by the user.")
                idx = start_idx_val + idx_offset
                filename = f"{idx:04d}.jpg"
                file_path = os.path.join(raw_dir, filename)
                url = src_info["url"]
                source = src_info["source"]
                
                coro = download_image(session, url, file_path, semaphore)
                task = asyncio.create_task(coro)
                active_tasks.append((filename, url, source, file_path, task))
                
            for filename, url, source, file_path, task in active_tasks:
                try:
                    # Check cancellation before processing downloaded file
                    if job_store.is_job_cancelled(job_id):
                        logger.info(f"Job {job_id} cancelled during download. Aborting.")
                        raise ValueError("Job was cancelled by the user.")
                    success = await task
                    if success:
                        downloaded_new += 1
                        metadata[filename] = {
                            "image_url": url,
                            "source_page": source,
                            "download_timestamp": datetime.utcnow().isoformat() + "Z",
                            "file_path": filename  # Relative path only!
                        }
                        if on_progress_update:
                            on_progress_update(len(metadata) - initial_metadata_count)
                        if (len(metadata) - initial_metadata_count) >= count:
                            logger.info(f"Target count of {count} reached. Stopping download.")
                            break
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.warning(f"Error downloading task {filename}: {e}")
        return len(metadata) - initial_metadata_count

    # First download pass
    start_idx = get_start_idx()
    downloaded_count = await run_downloads(image_sources, start_idx)

    # Retry scraping if we collected less than 50% of target
    if downloaded_count < int(count * 0.5):
        simplified = simplify_query(query)
        if simplified != query.strip().lower() and simplified:
            logger.info(f"Downloaded count {downloaded_count} is less than 50% of target {count}. Retrying scraping with simplified query: '{simplified}'")
            more_sources = []
            if api_key:
                try:
                    more_sources = await scrape_bing_api(simplified, count, api_key)
                except Exception:
                    pass
            if not more_sources:
                more_sources = await scrape_google_bing_playwright(simplified, count)
                
            # Deduplicate the retry sources
            existing_urls = {item["image_url"] for item in metadata.values()}
            new_sources = []
            seen_retry = set()
            for src in more_sources:
                u = src["url"]
                if u not in existing_urls and u not in seen_retry:
                    seen_retry.add(u)
                    new_sources.append(src)
                    
            if new_sources:
                logger.info(f"Found {len(new_sources)} additional candidate URLs. Starting retry downloads...")
                start_idx_retry = get_start_idx()
                downloaded_count = await run_downloads(new_sources, start_idx_retry)

    # Write metadata.json atomically
    metadata_file = os.path.join(raw_dir, "metadata.json")
    temp_metadata = metadata_file + ".tmp"
    with open(temp_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(temp_metadata, metadata_file)
    
    logger.info(f"Scraper finished. Total downloaded in this run: {downloaded_count} images.")
    return downloaded_count, metadata_file
