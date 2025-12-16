import random
from pathlib import Path

def create_test_corpus(num_lines: int = 500):
    
    if num_lines < 100:
        num_lines = 100
    elif num_lines > 1000:
        num_lines = 1000
    
    print(f"📚 Creating {num_lines} line test corpus...")
    
    templates = [
        "Artificial intelligence is transforming the modern world.",
        "Machine learning algorithms learn from large datasets.",
        "Deep learning neural networks are multi-layered structures.",
        "Natural language processing enables systems to understand human language.",
        "Computer vision systems can analyze images.",
        "Robotics technologies develop automated systems.",
        "Cloud computing offers scalable resources.",
        "Cybersecurity systems protect against malicious attacks.",
        "Blockchain technology uses distributed ledgers.",
        "Quantum computers can solve complex problems.",
        
        "The universe contains billions of galaxies.",
        "Photosynthesis is the process by which plants produce energy.",
        "Evolution explains the change of species over time.",
        "Climate change is a global problem.",
        "DNA encodes genetic information.",
        "Chemical reactions transform substances.",
        "Physics studies the fundamental laws of matter.",
        "Mathematics is used to model natural phenomena.",
        "Biology studies living organisms.",
        "Astronomy investigates celestial bodies.",
        
        "Language allows humans to communicate.",
        "Culture shapes the behaviors of societies.",
        "Education improves students' knowledge and skills.",
        "History records past events.",
        "Philosophy examines fundamental questions.",
        "Art expresses human emotions.",
        "Music unites humans.",
        "Literature tells human stories.",
        "Religion forms humans' belief systems.",
        "Politics guides societies' decision-making processes.",
        
        "Markets operate through supply and demand mechanisms.",
        "Innovation supports economic growth.",
        "Entrepreneurship transforms ideas into businesses.",
        "Management organizes resources.",
        "Marketing communicates value to customers.",
        "Finance manages money and investments.",
        "Accounting tracks financial operations.",
        "Operations optimize processes.",
        "Strategy makes long-term planning.",
        "Leadership guides teams.",
        
        "Medicine treats diseases.",
        "Nutrition is required for body functions.",
        "Exercise strengthens muscles.",
        "Sports affect physical and emotional well-being.",
        "Genetics affects susceptibility to various conditions.",
        "Immunology studies the body's defense system.",
        "Pharmacology develops drugs.",
        "Surgery repairs damaged tissues.",
        "Preventive care reduces risks.",
        "Public health protects communities.",
        
        "Breakfast is the most important meal of the day.",
        "Transportation allows humans to change places.",
        "Communication technology connects humans.",
        "Shopping meets daily needs.",
        "Entertainment fills humans' leisure time.",
        "Sports support a healthy life.",
        "Cooking is a creative art.",
        "Reading is the primary way to acquire knowledge.",
        "Travel allows discovering new cultures.",
        "Friendship is an important part of human life.",
        
        "Knowledge accumulates through observation and experience.",
        "Learning requires practice and repetition.",
        "Memory stores knowledge.",
        "Thinking is the process of processing knowledge.",
        "Creativity produces new ideas.",
        "Problem solving overcomes difficulties.",
        "Decision making evaluates options.",
        "Communication conveys ideas.",
        "Collaboration reaches common targets.",
        "Adaptation adapts to changing conditions.",
        
        "Programming languages express algorithms.",
        "Algorithms define step-by-step procedures.",
        "Data structures organize knowledge.",
        "Databases store large amounts of data.",
        "Networks connect computers.",
        "Operating systems manage hardware.",
        "Compilers translate source code.",
        "Debugging finds errors in programs.",
        "Testing verifies software meets requirements.",
        "Deployment releases software production.",
        
        "Turkish is a rich language.",
        "Turkey's cultural heritage is very valuable.",
        "Turkish cuisine is famous worldwide.",
        "Turkish literature has a rich past.",
        "Turkish music uses various instruments.",
        "Turkish history has a past of thousands of years.",
        "Turkish art combines traditional and modern elements.",
        "Turkish society is hospitable.",
        "Turkish economy is developing rapidly.",
        "Turkish technology produces innovative solutions.",
    ]
    
    sentence_patterns = [
        lambda: random.choice(templates),
        
        lambda: f"{random.choice(['How', 'Why', 'When', 'Where', 'Who'])} {random.choice(['works', 'happens', 'occurs', 'forms'])}?",
        
        lambda: f"{random.choice(templates)} Additionally, {random.choice(templates).lower()}",
        
        lambda: f"If we {random.choice(['understand', 'learn', 'apply'])}, then {random.choice(['we succeed', 'we make progress', 'results improve'])}.",
        
        lambda: f"{random.choice(['Traditional methods', 'Old approaches'])} differ from {random.choice(['modern solutions', 'new techniques'])}.",
        
        lambda: f"As {random.choice(['technology develops', 'knowledge increases', 'systems evolve'])}, {random.choice(['possibilities expand', 'opportunities increase', 'chances multiply'])}.",
        
        lambda: f"{random.choice(['Algorithm', 'Artificial intelligence', 'Machine learning'])} is a {random.choice(['computation method', 'system architecture', 'data structure'])}.",
        
        lambda: f"First {random.choice(['data is collected', 'models are trained', 'systems are tested'])}, then {random.choice(['results improve', 'performance increases', 'quality rises'])}.",
    ]
    
    corpus_lines = []
    
    for i in range(num_lines):
        pattern = random.choice(sentence_patterns)
        sentence = pattern()
        
        sentence = sentence.strip()
        
        if len(sentence) >= 10:
            corpus_lines.append(sentence)
    
    while len(corpus_lines) < num_lines:
        corpus_lines.extend(corpus_lines[:num_lines - len(corpus_lines)])
    
    corpus_lines = corpus_lines[:num_lines]
    
    output_path = Path("data/test_corpus.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Changed 'if' to 'f'
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in corpus_lines:
            f.write(line + '\n')
    
    file_size = output_path.stat().st_size / 1024
    print(f"✅ Test corpus created!")
    print(f"   📊 Line count: {len(corpus_lines):,}")
    print(f"   💾 File size: {file_size:.2f} KB")
    print(f"   📍 Location: {output_path.absolute()}")
    
    return output_path, len(corpus_lines), file_size

if __name__ == "__main__":
    print("=" * 60)
    print("📚 Test Corpus Generator (100-1000 lines)")
    print("=" * 60)
    print()
    
    output_path, num_lines, file_size = create_test_corpus(num_lines=500)
    
    print()
    print("=" * 60)
    print(f"✅ Corpus successfully created!")
    print(f"   📊 {num_lines:,} lines")
    print(f"   💾 {file_size:.2f} KB")
    print(f"   📍 {output_path.absolute()}")
    print()
    print("💡 You can use this corpus for training!")
    print("   Click the '🚀 Start Automatic Training' button in the Web UI.")
    print("=" * 60)