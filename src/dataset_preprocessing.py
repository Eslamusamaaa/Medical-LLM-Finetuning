import json
import re
from datasets import load_dataset
from tqdm import tqdm

class MedicalDataPreprocessor:
    def __init__(self):
        
        self.personal_keywords = [
            "اعاني", "عمري", "وزني", "دكتور", "ابني", "بنتي", "ابنتي", 
            "طولي", "عمليه", "حامل", "السن", "سني", "زوجي", "زوجتي", 
            "اخي", "اختي", "والدي", "والدتي", "امي", "ابي", "طفلي", "طفلتي"
        ]
        self.general_keywords = [
            "ما هي", "كيف", "اسباب", "اعراض", "علاج", "طرق", "هل", 
            "الفرق", "ما هو", "متي", "لماذا", "ماذا"
        ]

    def normalize_arabic(self, text):
        """Advanced Arabic text normalization."""
        
        text = re.sub(r"[\u064B-\u0652]", "", text)
        
        
        text = re.sub(r"[أإآ]", "ا", text)
        
        
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"ى", "ي", text)
        
        
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        
        return text

    def clean_text(self, text):
        """General text cleaning and noise removal."""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        
        text = re.sub(r'\n\s*\d+\s*(\n.*)?$', '', text, flags=re.DOTALL)
        text = re.sub(r'\n\s*(د\.|طاقم الطبي|الصيدلان|المصدر).*$', '', text, flags=re.DOTALL)
        
        
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def is_valid_pair(self, question, answer):
        """Validates if the QA pair is suitable for training."""
        if not question or not answer:
            return False
            
        
        if len(answer.split()) < 15 or len(question.split()) < 3:
            return False
            
        
        norm_q = self.normalize_arabic(question)
        
        
        has_personal = any(word in norm_q for word in self.personal_keywords)
        is_general = any(word in norm_q for word in self.general_keywords)
        
        return is_general and not has_personal

    def process_dataset(self, dataset_name, split="train"):
        print(f"Loading dataset: {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        
        processed_data = []
        seen_questions = set() 

        print("Starting preprocessing...")
        for example in tqdm(dataset):
            
            q = str(example.get("instruction") or example.get("question") or example.get("q_body") or "")
            a = str(example.get("output") or example.get("reply") or example.get("a_body") or "")
            
            if self.is_valid_pair(q, a):
                clean_q = self.clean_text(q)
                clean_a = self.clean_text(a)
                
                # Deduplication check
                if clean_q not in seen_questions:
                    processed_data.append({
                        "instruction": clean_q,
                        "input": "",
                        "output": clean_a
                    })
                    seen_questions.add(clean_q)
                    
        return processed_data

    def save_to_jsonl(self, data, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(data)} records to {filename}")

if __name__ == "__main__":
    preprocessor = MedicalDataPreprocessor()
    
    
    final_data = preprocessor.process_dataset("madilcy/arabic-medical-qa-MERGED-MAQA-MMMLU-MI")
    preprocessor.save_to_jsonl(final_data, "qwen_ready_medical_qa(2).jsonl")