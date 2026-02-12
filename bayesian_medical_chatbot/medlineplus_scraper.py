"""
MedlinePlus Web Scraper
Extracts detailed medical information from MedlinePlus pages.
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Optional
import re


class MedlinePlusScraper:
    """Scraper for extracting detailed disease information from MedlinePlus."""
    
    def __init__(self, delay: float = 1.5):
        """
        Initialize the scraper.
        
        Args:
            delay: Delay in seconds between requests (default: 1.5)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Medical Knowledge Base Builder (Educational/Research Purpose)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
    
    def scrape_disease(self, url: str, disease_name: str) -> Dict[str, str]:
        """
        Scrape detailed information for a disease from MedlinePlus.
        
        Args:
            url: MedlinePlus URL for the disease
            disease_name: Name of the disease (for logging)
            
        Returns:
            Dictionary with detailed disease information
        """
        print(f"Scraping {disease_name} from {url}...")
        
        try:
            # Fetch the page
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract information
            result = {
                'summary': self._extract_summary(soup),
                'symptoms_detailed': self._extract_section(soup, ['symptom', 'signs']),
                'causes': self._extract_section(soup, ['cause', 'risk factor']),
                'treatment': self._extract_section(soup, ['treatment', 'management', 'therapy']),
                'prevention': self._extract_section(soup, ['prevention', 'prevent']),
                'diagnosis': self._extract_section(soup, ['diagnos', 'test', 'exam']),
                'complications': self._extract_section(soup, ['complication', 'outlook', 'prognosis']),
                'when_to_see_doctor': self._extract_section(soup, ['when to', 'emergency', 'call your doctor', 'seek medical'])
            }
            
            # Clean up empty values
            for key in result:
                if not result[key] or result[key].strip() == '':
                    result[key] = 'Information not available on MedlinePlus page.'
            
            print(f"  ✓ Successfully scraped {disease_name}")
            
            # Respectful delay
            time.sleep(self.delay)
            
            return result
            
        except requests.RequestException as e:
            print(f"  ✗ Error fetching {disease_name}: {str(e)}")
            return self._get_empty_result()
        except Exception as e:
            print(f"  ✗ Error parsing {disease_name}: {str(e)}")
            return self._get_empty_result()
    
    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract the main summary/overview section."""
        # Try different possible summary locations
        summary = None
        
        # Method 1: Look for summary div
        summary_div = soup.find('div', {'id': 'topic-summary'})
        if summary_div:
            paragraphs = summary_div.find_all('p')
            if paragraphs:
                summary = ' '.join([p.get_text(strip=True) for p in paragraphs[:3]])  # First 3 paragraphs
        
        # Method 2: Look for main content area
        if not summary:
            main_content = soup.find('article') or soup.find('main') or soup.find('div', class_='main-content')
            if main_content:
                paragraphs = main_content.find_all('p', limit=3)
                summary = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        # Method 3: First few paragraphs in body
        if not summary:
            paragraphs = soup.find_all('p', limit=3)
            summary = ' '.join([p.get_text(strip=True) for p in paragraphs])
        
        return self._clean_text(summary) if summary else ''
    
    def _extract_section(self, soup: BeautifulSoup, keywords: list) -> str:
        """
        Extract content from a section matching given keywords.
        
        Args:
            soup: BeautifulSoup object
            keywords: List of keywords to match in headings
            
        Returns:
            Extracted text content
        """
        content_parts = []
        
        # Find all headings (h2, h3, h4)
        headings = soup.find_all(['h2', 'h3', 'h4'])
        
        for heading in headings:
            heading_text = heading.get_text(strip=True).lower()
            
            # Check if any keyword matches
            if any(keyword.lower() in heading_text for keyword in keywords):
                # Get content after this heading until next heading
                content = self._get_content_after_heading(heading)
                if content:
                    content_parts.append(content)
        
        # Combine all found content
        result = ' '.join(content_parts)
        return self._clean_text(result) if result else ''
    
    def _get_content_after_heading(self, heading) -> str:
        """Get all text content after a heading until the next heading."""
        content = []
        
        # Iterate through siblings until we hit another heading
        for sibling in heading.find_next_siblings():
            # Stop at next heading
            if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                break
            
            # Extract text from paragraphs and lists
            if sibling.name == 'p':
                text = sibling.get_text(strip=True)
                if text:
                    content.append(text)
            
            elif sibling.name in ['ul', 'ol']:
                items = sibling.find_all('li')
                for item in items:
                    text = item.get_text(strip=True)
                    if text:
                        content.append(f"• {text}")
            
            elif sibling.name == 'div':
                # Look for paragraphs and lists within div
                paragraphs = sibling.find_all('p')
                lists = sibling.find_all(['ul', 'ol'])
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text:
                        content.append(text)
                
                for lst in lists:
                    items = lst.find_all('li')
                    for item in items:
                        text = item.get_text(strip=True)
                        if text:
                            content.append(f"• {text}")
        
        return ' '.join(content)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?•\-()$/]', '', text)
        
        # Trim
        text = text.strip()
        
        return text
    
    def _get_empty_result(self) -> Dict[str, str]:
        """Return empty result structure."""
        return {
            'summary': 'Unable to retrieve information from MedlinePlus.',
            'symptoms_detailed': 'Information not available.',
            'causes': 'Information not available.',
            'treatment': 'Information not available.',
            'prevention': 'Information not available.',
            'diagnosis': 'Information not available.',
            'complications': 'Information not available.',
            'when_to_see_doctor': 'Information not available.'
        }
    
    def test_scraping(self, url: str):
        """Test scraping on a single URL and print results."""
        print(f"\n{'='*70}")
        print(f"Testing scraper on: {url}")
        print(f"{'='*70}\n")
        
        result = self.scrape_disease(url, "Test Disease")
        
        for key, value in result.items():
            print(f"\n{key.upper().replace('_', ' ')}:")
            print(f"{'-'*70}")
            print(f"{value[:500]}..." if len(value) > 500 else value)


def main():
    """Test the scraper with a sample URL."""
    scraper = MedlinePlusScraper()
    
    # Test with Malaria
    test_url = "https://medlineplus.gov/malaria.html"
    scraper.test_scraping(test_url)


if __name__ == "__main__":
    main()
