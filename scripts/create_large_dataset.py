import os
from pathlib import Path
import random
import re

def generate_large_corpus(output_path: str, num_lines: int = 50000):
    
    print(f"📚 Generating large corpus with {num_lines:,} lines...")
    
    templates = [
        "Artificial intelligence is revolutionizing {domain}.",
        "Machine learning algorithms can {action} with high accuracy.",
        "Deep neural networks consist of multiple {layers} layers.",
        "Natural language processing enables computers to understand {content}.",
        "Computer vision systems can identify and classify {objects}.",
        "Robotics combines {fields} to create autonomous systems.",
        "Cloud computing provides scalable {resources} for applications.",
        "Cybersecurity protects {systems} from malicious attacks.",
        "Blockchain technology ensures {property} through distributed ledgers.",
        "Quantum computing promises to solve {problems} efficiently.",
        
        "The universe contains billions of {objects} across vast distances.",
        "Photosynthesis converts {input} into {output} using sunlight.",
        "Evolution explains how {organisms} adapt to their environments.",
        "Climate change affects {systems} globally with long-term consequences.",
        "DNA encodes genetic information through {molecules} sequences.",
        "Chemical reactions transform {reactants} into {products}.",
        "Physics describes the fundamental {forces} governing matter.",
        "Mathematics provides tools for modeling {phenomena}.",
        "Biology studies living organisms and their {processes}.",
        "Astronomy explores celestial objects and their {properties}.",
        
        "Language enables humans to communicate {concepts} effectively.",
        "Culture shapes how people {behave} in different societies.",
        "Education develops {skills} and knowledge in students.",
        "History records past {events} for future understanding.",
        "Philosophy examines fundamental questions about {topics}.",
        "Art expresses human {emotions} through creative mediums.",
        "Music connects people through shared {experiences}.",
        "Literature captures human {stories} in written form.",
        "Religion provides frameworks for understanding {meaning}.",
        "Politics governs how societies make {decisions}.",
        
        "Markets allocate resources through supply and {demand} mechanisms.",
        "Innovation drives economic growth by creating new {solutions}.",
        "Entrepreneurship transforms ideas into viable {businesses}.",
        "Management coordinates {resources} to achieve organizational goals.",
        "Marketing communicates value to {customers} effectively.",
        "Finance manages money and investments for {returns}.",
        "Accounting tracks financial transactions and {performance}.",
        "Operations optimize processes for efficiency and {quality}.",
        "Strategy guides long-term planning and {direction}.",
        "Leadership inspires teams to achieve common {objectives}.",
        
        "Medicine treats diseases and improves human {health}.",
        "Nutrition provides essential nutrients for bodily {functions}.",
        "Exercise strengthens muscles and improves {fitness}.",
        "Mental health affects emotional and psychological {wellbeing}.",
        "Genetics influences susceptibility to various {conditions}.",
        "Immunology studies how the body defends against {pathogens}.",
        "Pharmacology develops drugs to treat {diseases}.",
        "Surgery repairs or removes damaged {tissues}.",
        "Preventive care reduces risk of future {problems}.",
        "Public health protects communities from {threats}.",
        
        "Knowledge accumulates through observation and {experimentation}.",
        "Learning requires practice and {repetition} to master skills.",
        "Memory stores information for future {retrieval}.",
        "Thinking involves processing information to reach {conclusions}.",
        "Creativity combines existing ideas into novel {solutions}.",
        "Problem-solving identifies and resolves {challenges}.",
        "Decision-making weighs options to choose {actions}.",
        "Communication transmits ideas between {individuals}.",
        "Collaboration combines efforts to achieve shared {goals}.",
        "Adaptation allows survival in changing {environments}.",
        
        "Programming languages express algorithms in executable {code}.",
        "Algorithms define step-by-step procedures for solving {problems}.",
        "Data structures organize information for efficient {access}.",
        "Databases store and retrieve large amounts of {data}.",
        "Networks connect computers to share {resources}.",
        "Operating systems manage hardware and software {resources}.",
        "Compilers translate source code into machine {instructions}.",
        "Debugging identifies and fixes errors in {programs}.",
        "Testing verifies that software meets specified {requirements}.",
        "Deployment releases software for production {use}.",
        
        "Truth represents accurate correspondence with {reality}.",
        "Beauty appeals to aesthetic {sensibilities}.",
        "Justice ensures fair treatment according to {principles}.",
        "Freedom allows individuals to make choices without {constraints}.",
        "Love connects people through deep emotional {bonds}.",
        "Wisdom combines knowledge with practical {understanding}.",
        "Courage enables action despite {fears}.",
        "Hope provides motivation to pursue {goals}.",
        "Faith involves trust beyond immediate {evidence}.",
        "Compassion drives caring actions toward {others}.",
    ]
    
    # Placeholder fills (logic preserved but names corrected)
    # Note: 'flls' was probably 'fills'
    fills = {
        "domain": ["healthcare", "education", "transportation", "communication", "entertainment"],
        "action": ["predict outcomes", "recognize patterns", "process information", "make decisions"],
        "layers": ["hidden", "intermediate", "output", "input"],
        "content": ["human language", "sentences", "meaning", "context"],
        "objects": ["images", "shapes", "patterns", "scenes"],
        "fields": ["engineering", "computing", "sensors"],
        "resources": ["storage", "processing", "networking"],
        "systems": ["networks", "data", "information"],
        "property": ["transparency", "security", "immutability"],
        "problems": ["complex optimization", "cryptography", "simulation"],
    }
    
    corpus_lines = []
    
    for i in range(num_lines):
        pattern = random.choice(range(10))
        
        if pattern == 0:
            template = random.choice(templates)
            sentence = re.sub(r'\{[^}]+\}', lambda m: random.choice(['data', 'information', 'systems', 'technology', 'knowledge']), template)
        elif pattern == 1:
            t1 = random.choice(templates)
            t2 = random.choice(templates)
            s1 = re.sub(r'\{[^}]+\}', lambda m: random.choice(['data', 'information', 'systems']), t1)
            s2 = re.sub(r'\{[^}]+\}', lambda m: random.choice(['data', 'information', 'systems']), t2)
            sentence = f"{s1} Additionally, {s2.lower()}"
        elif pattern == 2:
            sentence = f"How does {random.choice(['technology', 'science', 'society'])} affect {random.choice(['people', 'systems', 'environments'])}?"
        elif pattern == 3:
            sentence = f"If {random.choice(['we understand', 'systems learn', 'people adapt'])}, then {random.choice(['progress accelerates', 'efficiency improves', 'outcomes enhance'])}."
        elif pattern == 4:
            sentence = f"{random.choice(['Traditional methods', 'Previous approaches', 'Older techniques'])} differ from {random.choice(['modern solutions', 'current practices', 'recent innovations'])}."
        elif pattern == 5:
            sentence = f"Because {random.choice(['technology advances', 'knowledge increases', 'systems evolve'])}, {random.choice(['capabilities expand', 'possibilities grow', 'opportunities emerge'])}."
        elif pattern == 6:
            sentence = f"{random.choice(['Complex systems', 'Modern technologies', 'Advanced algorithms'])} exhibit {random.choice(['emergent properties', 'scalable behavior', 'adaptive responses'])}."
        elif pattern == 7:
            sentence = f"First, {random.choice(['data is collected', 'models are trained', 'systems are tested'])}, then {random.choice(['patterns emerge', 'results improve', 'performance increases'])}."
        elif pattern == 8:
            term = random.choice(['algorithm', 'neural network', 'language model', 'transformer', 'embedding'])
            sentence = f"A {term} is a {random.choice(['computational method', 'system architecture', 'data structure'])} that {random.choice(['processes information', 'learns patterns', 'generates outputs'])}."
        else:
            template = random.choice(templates)
            sentence = re.sub(r'\{[^}]+\}', lambda m: random.choice(['data', 'information', 'systems', 'technology', 'knowledge']), template)
        
        sentence = sentence.strip()
        
        if len(sentence) > 10:
            corpus_lines.append(sentence)
    
    while len(corpus_lines) < num_lines:
        corpus_lines.extend(corpus_lines[:num_lines - len(corpus_lines)])
    
    corpus_lines = corpus_lines[:num_lines]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in corpus_lines:
            f.write(line + '\n')
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Generated corpus: {len(corpus_lines):,} lines, {file_size:.2f} MB")
    print(f"📍 Saved to: {output_path.absolute()}")
    
    return len(corpus_lines), file_size

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    corpus_path = project_root / "data" / "sample_corpus.txt"
    
    print("🚀 Creating large training corpus...")
    print("=" * 60)
    
    num_lines, size = generate_large_corpus(str(corpus_path), num_lines=200000)
    
    print("=" * 60)
    print(f"✅ Corpus generation complete!")
    print(f"   Lines: {num_lines:,}")
    print(f"   Size: {size:.2f} MB")
    print(f"   Location: {corpus_path.absolute()}")