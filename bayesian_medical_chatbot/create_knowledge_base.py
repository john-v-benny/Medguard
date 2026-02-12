"""
Comprehensive Knowledge Base Creator for Bayesian Medical Chatbot
Scrapes MedlinePlus for all 41 diseases in the Bayesian network.
"""

import csv
import pickle
from datetime import datetime
from typing import List, Dict
from medlineplus_scraper import MedlinePlusScraper
import time

# Disease name mappings: Bayesian name -> MedlinePlus URL
DISEASE_MAPPINGS = {
    'AIDS': 'https://medlineplus.gov/hivaids.html',
    'Acne': 'https://medlineplus.gov/acne.html',
    'Alcoholic hepatitis': 'https://medlineplus.gov/alcoholicliverdisease.html',
    'Allergy': 'https://medlineplus.gov/allergy.html',
    'Arthritis': 'https://medlineplus.gov/arthritis.html',
    'Bronchial Asthma': 'https://medlineplus.gov/asthma.html',
    'Cervical spondylosis': 'https://medlineplus.gov/cervicalspondylosis.html',
    'Chicken pox': 'https://medlineplus.gov/chickenpox.html',
    'Chronic cholestasis': 'https://medlineplus.gov/liverdiseases.html',
    'Common Cold': 'https://medlineplus.gov/commoncold.html',
    'Dengue': 'https://medlineplus.gov/dengue.html',
    'Diabetes': 'https://medlineplus.gov/diabetes.html',
    'Dimorphic hemmorhoids(piles)': 'https://medlineplus.gov/hemorrhoids.html',
    'Drug Reaction': 'https://medlineplus.gov/drugreactions.html',
    'Fungal infection': 'https://medlineplus.gov/fungalinfections.html',
    'GERD': 'https://medlineplus.gov/gerd.html',
    'Gastroenteritis': 'https://medlineplus.gov/gastroenteritis.html',
    'Heart attack': 'https://medlineplus.gov/heartattack.html',
    'Hepatitis A': 'https://medlineplus.gov/hepatitisa.html',
    'Hepatitis B': 'https://medlineplus.gov/hepatitisb.html',
    'Hepatitis C': 'https://medlineplus.gov/hepatitisc.html',
    'Hepatitis D': 'https://medlineplus.gov/hepatitisd.html',
    'Hepatitis E': 'https://medlineplus.gov/hepatitise.html',
    'hepatitis A': 'https://medlineplus.gov/hepatitisa.html',  # Duplicate with different case
    'Hypertension': 'https://medlineplus.gov/highbloodpressure.html',
    'Hyperthyroidism': 'https://medlineplus.gov/hyperthyroidism.html',
    'Hypoglycemia': 'https://medlineplus.gov/hypoglycemia.html',
    'Hypothyroidism': 'https://medlineplus.gov/hypothyroidism.html',
    'Impetigo': 'https://medlineplus.gov/impetigo.html',
    'Jaundice': 'https://medlineplus.gov/jaundice.html',
    'Malaria': 'https://medlineplus.gov/malaria.html',
    'Migraine': 'https://medlineplus.gov/migraine.html',
    'Osteoarthristis': 'https://medlineplus.gov/osteoarthritis.html',
    'Paralysis (brain hemorrhage)': 'https://medlineplus.gov/stroke.html',
    'Peptic ulcer diseae': 'https://medlineplus.gov/pepticulcer.html',
    'Pneumonia': 'https://medlineplus.gov/pneumonia.html',
    'Psoriasis': 'https://medlineplus.gov/psoriasis.html',
    'Tuberculosis': 'https://medlineplus.gov/tuberculosis.html',
    'Typhoid': 'https://medlineplus.gov/typhoidfever.html',
    'Urinary tract infection': 'https://medlineplus.gov/urinarytractinfections.html',
    'Varicose veins': 'https://medlineplus.gov/varicoseveins.html',
    '(vertigo) Paroymsal  Positional Vertigo': 'https://medlineplus.gov/dizzinessandvertigo.html',
}


def load_bayesian_diseases() -> List[str]:
    """Load disease names from Bayesian network model."""
    with open('models/disease_bayesian_network.pkl', 'rb') as f:
        data = pickle.load(f)
    return sorted(data['diseases'])


def create_knowledge_base(use_scraper: bool = True) -> List[Dict]:
    """
    Create comprehensive knowledge base for all 41 diseases.
    
    Args:
        use_scraper: If True, scrapes MedlinePlus. If False, uses basic info.
        
    Returns:
        List of disease dictionaries with complete information
    """
    diseases = load_bayesian_diseases()
    knowledge_base = []
    
    print(f"\n{'='*80}")
    print(f"Creating Knowledge Base for {len(diseases)} Diseases")
    print(f"{'='*80}\n")
    
    if use_scraper:
        scraper = MedlinePlusScraper(delay=2.0)  # 2 second delay to be respectful
    
    for i, disease in enumerate(diseases, 1):
        print(f"\n[{i}/{len(diseases)}] Processing: {disease}")
        print("-" * 80)
        
        # Get MedlinePlus URL
        url = DISEASE_MAPPINGS.get(disease)
        
        if not url:
            print(f"  ⚠ No URL mapping found for '{disease}' - using generic info")
            disease_data = create_generic_entry(disease)
        elif use_scraper:
            # Scrape detailed information
            scraped_data = scraper.scrape_disease(url, disease)
            disease_data = {
                'disease': disease,
                'description': scraped_data['summary'],
                'symptoms': scraped_data['symptoms_detailed'],
                'causes': scraped_data['causes'],
                'treatment': scraped_data['treatment'],
                'medications': extract_medications(scraped_data['treatment']),
                'diagnosis': scraped_data['diagnosis'],
                'prevention': scraped_data['prevention'],
                'complications': scraped_data['complications'],
                'when_to_see_doctor': scraped_data['when_to_see_doctor'],
                'source_url': url,
                'source_name': 'MedlinePlus - National Library of Medicine (NLM)',
                'last_updated': datetime.now().strftime('%Y-%m-%d')
            }
        else:
            disease_data = create_generic_entry(disease, url)
        
        knowledge_base.append(disease_data)
        print(f"  ✓ Added {disease} to knowledge base")
    
    print(f"\n{'='*80}")
    print(f"Knowledge Base Creation Complete: {len(knowledge_base)} diseases")
    print(f"{'='*80}\n")
    
    return knowledge_base


def extract_medications(treatment_text: str) -> str:
    """
    Extract medication information from treatment text.
    
    Args:
        treatment_text: Full treatment description
        
    Returns:
        Extracted medication information
    """
    if not treatment_text or treatment_text == 'Information not available.':
        return 'Consult a healthcare professional for medication recommendations.'
    
    # Look for medication-related keywords
    med_keywords = ['medicine', 'medication', 'drug', 'antibiotic', 'prescription', 'pill', 'tablet']
    
    sentences = treatment_text.split('.')
    med_sentences = []
    
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in med_keywords):
            med_sentences.append(sentence.strip())
    
    if med_sentences:
        return '. '.join(med_sentences[:3]) + '.'  # First 3 relevant sentences
    else:
        return 'Medication information included in treatment plan. Consult a healthcare professional.'


def create_generic_entry(disease: str, url: str = None) -> Dict:
    """Create a generic entry for diseases without detailed scraping."""
    return {
        'disease': disease,
        'description': f'{disease} is a medical condition. For detailed information, please consult a healthcare professional.',
        'symptoms': 'Symptoms vary. Consult a healthcare professional for accurate diagnosis.',
        'causes': 'Multiple factors may contribute to this condition.',
        'treatment': 'Treatment options are available. Consult a healthcare professional for personalized treatment plan.',
        'medications': 'Consult a healthcare professional for medication recommendations.',
        'diagnosis': 'Professional medical diagnosis is required.',
        'prevention': 'Preventive measures vary. Consult a healthcare professional.',
        'complications': 'Complications may occur if left untreated.',
        'when_to_see_doctor': 'Seek medical attention if you experience concerning symptoms.',
        'source_url': url or 'https://medlineplus.gov/',
        'source_name': 'MedlinePlus - National Library of Medicine (NLM)',
        'last_updated': datetime.now().strftime('%Y-%m-%d')
    }


def save_to_csv(data: List[Dict], filename: str = 'disease_knowledge_base.csv'):
    """Save knowledge base to CSV file."""
    if not data:
        print("No data to save!")
        return
    
    fieldnames = [
        'disease', 'description', 'symptoms', 'causes', 'treatment',
        'medications', 'diagnosis', 'prevention', 'complications',
        'when_to_see_doctor', 'source_url', 'source_name', 'last_updated'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\n✓ Knowledge base saved to: {filename}")
    print(f"  Total diseases: {len(data)}")
    print(f"  Fields per disease: {len(fieldnames)}")


def main():
    """Main function to create the knowledge base."""
    print("\n" + "="*80)
    print("BAYESIAN MEDICAL CHATBOT - KNOWLEDGE BASE CREATOR")
    print("="*80)
    
    # Ask user if they want to scrape or use generic data
    print("\nOptions:")
    print("1. Scrape detailed information from MedlinePlus (recommended, ~5-10 minutes)")
    print("2. Use generic information (fast, but less detailed)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    use_scraper = choice == '1'
    
    if use_scraper:
        print("\n⚠ This will make ~41 requests to MedlinePlus.")
        print("  The scraper includes respectful delays (2 seconds between requests).")
        print("  Estimated time: 5-10 minutes")
        confirm = input("\nProceed? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    # Create knowledge base
    knowledge_base = create_knowledge_base(use_scraper=use_scraper)
    
    # Save to CSV
    save_to_csv(knowledge_base)
    
    print("\n" + "="*80)
    print("KNOWLEDGE BASE CREATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review disease_knowledge_base.csv")
    print("2. Run: python vector_db_setup.py")
    print("3. Test RAG integration")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
