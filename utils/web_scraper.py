import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, List, Tuple
from urllib.parse import urlparse, urljoin
import time
import logging

logger = logging.getLogger('qkv_core')

class WebScraper:
    
    def __init__(self, timeout: int = 30, max_content_length: int = 100000):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_url(self, text: str) -> Optional[str]:
        # Regex pattern for extracting URLs (fixed: \is -> \s)
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.findall(url_pattern, text)
        
        if matches:
            for url in matches:
                try:
                    parsed = urlparse(url)
                    # Fixed: clearloc -> netloc
                    if parsed.scheme in ['http', 'https'] and parsed.netloc:
                    return url
                except:
                    continue
        
        return None
    
    def fetch_content(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            # Fixed: f-string syntax (if" -> f")
            logger.info(f"Fetching content from URL: {url}")
            
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return None, "Invalid URL format"
            
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            # Fixed: plan -> plain, note -> not
            if 'text/html' not in content_type and 'text/plain' not in content_type:
                return None, f"Unsupported content type: {content_type}"
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Fixed: scrpt -> script, asde -> aside
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            text = soup.get_text()
            
            # Fixed: spltlines -> splitlines, strp -> strip
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Fixed: \in -> \n (newline)
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            if len(text) > self.max_content_length:
                text = text[:self.max_content_length]
                # chmddlecters -> characters
                logger.warning(f"Content truncated to {self.max_content_length} characters")
            
            logger.info(f"Successfully extracted {len(text)} characters from {url}")
            return text, None
            
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout for {url}"
            logger.error(error_msg)
            return None, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error for {url}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Error processing {url}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return None, error_msg
    
    def extract_links(self, url: str, base_url: Optional[str] = None) -> List[str]:
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            links = []
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                absolute_url = urljoin(url, href)
                
                parsed = urlparse(absolute_url)
                if parsed.scheme in ['http', 'https']:
                    links.append(absolute_url)
            
            return links
            
        except Exception as e:
            logger.error(f"Error extracting links from {url}: {str(e)}")
            return []

def clean_text_for_training(text: str) -> str:
    # Normalize whitespace: multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Preserve Turkish characters (çğöşüÇĞÖŞÜ) and common punctuation
    # \w in Python already includes unicode characters, but we explicitly preserve
    # Turkish characters to ensure proper text processing
    text = re.sub(r'[^\w\sçğöşüÇĞÖŞÜ.,!?;:()\[\]{}\'"-]', '', text)
    
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
    
    text = '\n'.join(lines)
    
    if len(text) > 50000:
        text = text[:50000]
    
    return text.strip()