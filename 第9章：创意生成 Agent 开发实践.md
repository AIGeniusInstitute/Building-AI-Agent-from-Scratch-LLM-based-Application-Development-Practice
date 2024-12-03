# 第9章：创意生成 Agent 开发实践

创意生成 Agent 是一种能够在各种领域产生新颖、有价值想法的 AI 系统。本章将详细介绍创意生成 Agent 的开发过程，包括系统设计、知识库构建、生成技术、评估方法等方面。

## 9.1 创意生成系统设计

### 9.1.1 创意领域定义

明确定义创意生成的目标领域是系统设计的第一步。这涉及确定具体的应用场景、目标受众和期望输出。

示例（创意领域定义类）：

```python
class CreativeDomain:
    def __init__(self, name, description, target_audience, output_types):
        self.name = name
        self.description = description
        self.target_audience = target_audience
        self.output_types = output_types

    def __str__(self):
        return f"Creative Domain: {self.name}\n" \
               f"Description: {self.description}\n" \
               f"Target Audience: {', '.join(self.target_audience)}\n" \
               f"Output Types: {', '.join(self.output_types)}"

# 使用示例
product_design_domain = CreativeDomain(
    name="Product Design",
    description="Generate innovative product concepts for everyday consumer goods",
    target_audience=["Young adults", "Urban professionals"],
    output_types=["Product concept", "Sketch", "Feature list"]
)

print(product_design_domain)
```

### 9.1.2 生成流程设计

设计一个结构化的创意生成流程，包括灵感收集、概念生成、评估和精炼等阶段。

示例（创意生成流程类）：

```python
from enum import Enum

class CreativeStage(Enum):
    INSPIRATION = 1
    IDEATION = 2
    EVALUATION = 3
    REFINEMENT = 4

class CreativeProcess:
    def __init__(self, domain):
        self.domain = domain
        self.stages = [
            (CreativeStage.INSPIRATION, self.gather_inspiration),
            (CreativeStage.IDEATION, self.generate_concepts),
            (CreativeStage.EVALUATION, self.evaluate_ideas),
            (CreativeStage.REFINEMENT, self.refine_concepts)
        ]

    def gather_inspiration(self):
        print("Gathering inspiration from various sources...")
        # 实现灵感收集逻辑

    def generate_concepts(self):
        print("Generating initial concepts...")
        # 实现概念生成逻辑

    def evaluate_ideas(self):
        print("Evaluating generated ideas...")
        # 实现创意评估逻辑

    def refine_concepts(self):
        print("Refining selected concepts...")
        # 实现概念精炼逻辑

    def execute(self):
        for stage, function in self.stages:
            print(f"\nExecuting stage: {stage.name}")
            function()

# 使用示例
creative_process = CreativeProcess(product_design_domain)
creative_process.execute()
```

### 9.1.3 评估指标确立

建立一套全面的评估指标，用于衡量生成创意的质量、新颖性和实用性。

示例（创意评估指标类）：

```python
class CreativeEvaluationMetrics:
    def __init__(self):
        self.metrics = {
            "novelty": self.evaluate_novelty,
            "usefulness": self.evaluate_usefulness,
            "feasibility": self.evaluate_feasibility,
            "relevance": self.evaluate_relevance
        }

    def evaluate_novelty(self, idea):
        # 实现新颖性评估逻辑
        return random.uniform(0, 1)  # 示例：返回0到1之间的随机值

    def evaluate_usefulness(self, idea):
        # 实现实用性评估逻辑
        return random.uniform(0, 1)

    def evaluate_feasibility(self, idea):
        # 实现可行性评估逻辑
        return random.uniform(0, 1)

    def evaluate_relevance(self, idea):
        # 实现相关性评估逻辑
        return random.uniform(0, 1)

    def evaluate_idea(self, idea):
        results = {}
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(idea)
        return results

# 使用示例
evaluator = CreativeEvaluationMetrics()
idea = "A self-cleaning water bottle that purifies water using UV light"
evaluation_results = evaluator.evaluate_idea(idea)

print("Idea Evaluation Results:")
for metric, score in evaluation_results.items():
    print(f"{metric.capitalize()}: {score:.2f}")
```

## 9.2 灵感源与知识库构建

### 9.2.1 多源数据采集

从多种来源收集数据，以构建丰富的知识库作为创意生成的基础。

示例（多源数据采集器）：

```python
import requests
from bs4 import BeautifulSoup
import tweepy

class DataCollector:
    def __init__(self, config):
        self.config = config

    def collect_web_data(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 实现网页数据提取逻辑
        return soup.get_text()

    def collect_social_media_data(self, query):
        auth = tweepy.OAuthHandler(self.config['twitter_api_key'], self.config['twitter_api_secret'])
        auth.set_access_token(self.config['twitter_access_token'], self.config['twitter_access_token_secret'])
        api = tweepy.API(auth)

        tweets = api.search_tweets(q=query, count=100)
        return [tweet.text for tweet in tweets]

    def collect_academic_papers(self, topic):
        # 实现学术论文收集逻辑（可能需要特定API或爬虫）
        pass

# 使用示例
config = {
    'twitter_api_key': 'your_api_key',
    'twitter_api_secret': 'your_api_secret',
    'twitter_access_token': 'your_access_token',
    'twitter_access_token_secret': 'your_access_token_secret'
}

collector = DataCollector(config)

web_data = collector.collect_web_data('https://example.com')
social_media_data = collector.collect_social_media_data('innovation')

print(f"Collected {len(web_data)} characters of web data")
print(f"Collected {len(social_media_data)} social media posts")
```

### 9.2.2 创意元素提取

从收集的数据中提取关键的创意元素，如概念、属性、关系等。

示例（创意元素提取器）：

```python
import spacy
from collections import Counter

class CreativeElementExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_keywords(self, text):
        doc = self.nlp(text)
        keywords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return Counter(keywords).most_common(10)

    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return Counter(entities).most_common(10)

    def extract_noun_chunks(self, text):
        doc = self.nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        return Counter(noun_chunks).most_common(10)

# 使用示例
extractor = CreativeElementExtractor()

text = "Apple's innovative design combines sleek aesthetics with cutting-edge technology, revolutionizing the smartphone industry."

keywords = extractor.extract_keywords(text)
entities = extractor.extract_entities(text)
noun_chunks = extractor.extract_noun_chunks(text)

print("Keywords:", keywords)
print("Entities:", entities)
print("Noun Chunks:", noun_chunks)
```

### 9.2.3 知识图谱构建

将提取的创意元素组织成知识图谱，以捕捉概念之间的关系。

示例（简单的知识图谱构建器）：

```python
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_concept(self, concept):
        self.graph.add_node(concept)

    def add_relation(self, concept1, relation, concept2):
        self.graph.add_edge(concept1, concept2, relation=relation)

    def visualize(self):
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(12, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        plt.title("Creative Knowledge Graph")
        plt.axis('off')
        plt.show()

# 使用示例
kg_builder = KnowledgeGraphBuilder()

# 添加概念和关系
kg_builder.add_concept("Smartphone")
kg_builder.add_concept("Touchscreen")
kg_builder.add_concept("Battery")
kg_builder.add_concept("Camera")

kg_builder.add_relation("Smartphone", "has", "Touchscreen")
kg_builder.add_relation("Smartphone", "contains", "Battery")
kg_builder.add_relation("Smartphone", "features", "Camera")

# 可视化知识图谱
kg_builder.visualize()
```

通过这些组件，创意生成 Agent 可以构建丰富的知识基础，为后续的创意生成过程提供充足的素材和灵感。在实际应用中，这些组件可能需要更复杂的实现，包括更高级的自然语言处理技术、更全面的数据源集成，以及更复杂的知识表示方法。此外，还需要考虑数据的质量控制、更新机制，以及如何有效地利用构建的知识库进行创意生成。

## 9.3 LLM 创意生成技术

利用大语言模型（LLM）进行创意生成是当前最先进的方法之一。这种方法可以产生高质量、多样化的创意输出。

### 9.3.1 条件生成方法

条件生成允许我们通过提供特定的输入条件来引导 LLM 的创意输出。

示例（使用 GPT-3 进行条件创意生成）：

```python
import openai

class ConditionalCreativeGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate(self, prompt, conditions, max_tokens=100):
        formatted_prompt = f"{prompt}\n\nConditions: {', '.join(conditions)}\n\nCreative output:"
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.8,
        )
        
        return response.choices[0].text.strip()

# 使用示例
generator = ConditionalCreativeGenerator("your-api-key-here")

prompt = "Design a new type of transportation device"
conditions = ["eco-friendly", "urban environment", "compact"]

creative_output = generator.generate(prompt, conditions)
print(f"Creative Output: {creative_output}")
```

### 9.3.2 风格迁移技术

风格迁移技术允许我们将一种风格或领域的特征应用到另一个领域的创意中。

示例（使用 GPT-3 进行简单的风格迁移）：

```python
class StyleTransferGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_with_style(self, content, style, max_tokens=150):
        prompt = f"Transform the following content into the style of {style}:\n\nContent: {content}\n\nStyled output:"
        
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )
        
        return response.choices[0].text.strip()

# 使用示例
style_generator = StyleTransferGenerator("your-api-key-here")

content = "A device that helps people stay hydrated throughout the day"
style = "cyberpunk science fiction"

styled_output = style_generator.generate_with_style(content, style)
print(f"Styled Output: {styled_output}")
```

### 9.3.3 多样性增强策略

为了避免创意输出变得单调或重复，我们可以实施多样性增强策略。

示例（使用 GPT-3 的多样性参数和多次生成来增强多样性）：

```python
import numpy as np

class DiverseCreativeGenerator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_diverse(self, prompt, num_outputs=3, temperature_range=(0.5, 1.0), max_tokens=100):
        outputs = []
        
        for _ in range(num_outputs):
            temperature = np.random.uniform(*temperature_range)
            
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=temperature,
            )
            
            outputs.append(response.choices[0].text.strip())
        
        return outputs

# 使用示例
diverse_generator = DiverseCreativeGenerator("your-api-key-here")

prompt = "Invent a new kitchen gadget that solves a common cooking problem"

diverse_outputs = diverse_generator.generate_diverse(prompt)
for i, output in enumerate(diverse_outputs, 1):
    print(f"Output {i}: {output}\n")
```

## 9.4 创意评估与筛选

生成创意后，需要对其进行评估和筛选，以确保输出的质量和相关性。

### 9.4.1 新颖性评估

评估创意的新颖性，确保它们不仅仅是现有想法的重复。

示例（使用 Word2Vec 进行简单的新颖性评估）：

```python
import gensim.downloader as api
from scipy.spatial.distance import cosine

class NoveltyEvaluator:
    def __init__(self):
        self.model = api.load("word2vec-google-news-300")
        self.existing_ideas = []  # 假设这是已知的想法列表

    def calculate_novelty(self, new_idea):
        if not self.existing_ideas:
            return 1.0  # 如果没有现有想法，则认为是完全新颖的
        
        max_similarity = 0
        for existing_idea in self.existing_ideas:
            similarity = self.semantic_similarity(new_idea, existing_idea)
            max_similarity = max(max_similarity, similarity)
        
        return 1 - max_similarity

    def semantic_similarity(self, idea1, idea2):
        vec1 = self.get_idea_vector(idea1)
        vec2 = self.get_idea_vector(idea2)
        return 1 - cosine(vec1, vec2)

    def get_idea_vector(self, idea):
        words = idea.lower().split()
        vectors = [self.model[word] for word in words if word in self.model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.model.vector_size)

# 使用示例
novelty_evaluator = NoveltyEvaluator()
novelty_evaluator.existing_ideas = [
    "A smart refrigerator that tracks food expiration",
    "A voice-controlled coffee maker"
]

new_idea = "A holographic cooking assistant that guides you through recipes"
novelty_score = novelty_evaluator.calculate_novelty(new_idea)
print(f"Novelty score for '{new_idea}': {novelty_score:.2f}")
```

### 9.4.2 实用性分析

评估创意的实用性和可行性，确保它们能够解决实际问题或满足用户需求。

示例（使用简单的规则基础评分系统）：

```python
class UtilityAnalyzer:
    def __init__(self):
        self.criteria = {
            "solves_problem": 0.4,
            "ease_of_use": 0.3,
            "cost_effectiveness": 0.2,
            "scalability": 0.1
        }

    def analyze_utility(self, idea, ratings):
        if set(ratings.keys()) != set(self.criteria.keys()):
            raise ValueError("Ratings must include all criteria")

        utility_score = sum(self.criteria[criterion] * ratings[criterion] for criterion in self.criteria)
        return utility_score

# 使用示例
utility_analyzer = UtilityAnalyzer()

idea = "A smart water bottle that reminds you to drink water and tracks your hydration levels"
ratings = {
    "solves_problem": 0.8,
    "ease_of_use": 0.7,
    "cost_effectiveness": 0.6,
    "scalability": 0.9
}

utility_score = utility_analyzer.analyze_utility(idea, ratings)
print(f"Utility score for '{idea}': {utility_score:.2f}")
```

### 9.4.3 市场潜力预测

评估创意的市场潜力，包括目标受众规模、竞争格局和潜在收入等因素。

示例（简单的市场潜力评估模型）：

```python
import numpy as np

class MarketPotentialPredictor:
    def __init__(self):
        self.factors = {
            "target_audience_size": 0.3,
            "competition_level": 0.2,
            "innovation_degree": 0.2,
            "implementation_cost": 0.15,
            "revenue_potential": 0.15
        }

    def predict_potential(self, idea, factor_scores):
        if set(factor_scores.keys()) != set(self.factors.keys()):
            raise ValueError("Factor scores must include all factors")

        # 将竞争水平和实施成本转换为正面指标
        factor_scores['competition_level'] = 1 - factor_scores['competition_level']
        factor_scores['implementation_cost'] = 1 - factor_scores['implementation_cost']

        weighted_scores = [self.factors[factor] * score for factor, score in factor_scores.items()]
        potential_score = np.mean(weighted_scores)

        return potential_score

# 使用示例
market_predictor = MarketPotentialPredictor()

idea = "A personalized AI-powered meal planning and grocery shopping assistant"
factor_scores = {
    "target_audience_size": 0.8,
    "competition_level": 0.6,  # 0.6 表示中等竞争水平
    "innovation_degree": 0.7,
    "implementation_cost": 0.5,  # 0.5 表示中等实施成本
    "revenue_potential": 0.75
}

market_potential = market_predictor.predict_potential(idea, factor_scores)
print(f"Market potential score for '{idea}': {market_potential:.2f}")
```

通过这些评估和筛选机制，创意生成 Agent 可以更好地识别和优先考虑高质量、高潜力的创意。在实际应用中，这些评估方法可能需要更复杂的实现，可能涉及机器学习模型、专家系统或更全面的市场分析工具。此外，还需要考虑如何平衡不同的评估指标，以及如何根据特定领域或项目的需求调整评估标准。

## 9.5 人机协作创意优化

人机协作是提升创意质量和实用性的有效方式。通过结合人类的直觉和经验与AI的计算能力，可以产生更加优秀的创意成果。

### 9.5.1 反馈收集机制

设计一个有效的反馈收集系统，使人类专家能够轻松评价和改进AI生成的创意。

示例（反馈收集系统）：

```python
class FeedbackCollector:
    def __init__(self):
        self.feedback_categories = ["novelty", "usefulness", "feasibility", "market_potential"]
        self.feedback_data = []

    def collect_feedback(self, idea, expert_id):
        print(f"Idea: {idea}")
        feedback = {}
        for category in self.feedback_categories:
            while True:
                try:
                    score = float(input(f"Rate {category} (0-10): "))
                    if 0 <= score <= 10:
                        feedback[category] = score
                        break
                    else:
                        print("Please enter a number between 0 and 10.")
                except ValueError:
                    print("Please enter a valid number.")
        
        comments = input("Additional comments: ")
        feedback["comments"] = comments
        feedback["expert_id"] = expert_id
        
        self.feedback_data.append({"idea": idea, "feedback": feedback})
        return feedback

    def get_average_scores(self, idea):
        relevant_feedback = [item["feedback"] for item in self.feedback_data if item["idea"] == idea]
        if not relevant_feedback:
            return None
        
        avg_scores = {}
        for category in self.feedback_categories:
            scores = [feedback[category] for feedback in relevant_feedback]
            avg_scores[category] = sum(scores) / len(scores)
        
        return avg_scores

# 使用示例
collector = FeedbackCollector()

idea = "A smart garden that automatically adjusts watering based on plant needs and weather forecasts"
feedback = collector.collect_feedback(idea, expert_id="expert1")
print(f"Collected feedback: {feedback}")

avg_scores = collector.get_average_scores(idea)
if avg_scores:
    print(f"Average scores for the idea: {avg_scores}")
```

### 9.5.2 交互式创意迭代

实现一个交互式系统，允许人类专家和AI系统进行多轮创意优化。

示例（交互式创意迭代系统）：

```python
class InteractiveCreativeIterator:
    def __init__(self, creative_generator, feedback_collector):
        self.generator = creative_generator
        self.collector = feedback_collector
        self.iteration_history = []

    def iterate(self, initial_idea, max_iterations=3):
        current_idea = initial_idea
        for i in range(max_iterations):
            print(f"\nIteration {i+1}:")
            print(f"Current idea: {current_idea}")
            
            # 收集人类反馈
            feedback = self.collector.collect_feedback(current_idea, f"expert_{i}")
            
            # 基于反馈生成改进的创意
            improvement_prompt = f"Improve the following idea based on this feedback: {feedback}\n\nIdea: {current_idea}\n\nImproved idea:"
            improved_idea = self.generator.generate(improvement_prompt)
            
            self.iteration_history.append({
                "iteration": i+1,
                "original_idea": current_idea,
                "feedback": feedback,
                "improved_idea": improved_idea
            })
            
            current_idea = improved_idea
            
            continue_iterating = input("Continue iterating? (y/n): ").lower() == 'y'
            if not continue_iterating:
                break
        
        return current_idea, self.iteration_history

# 使用示例
creative_generator = ConditionalCreativeGenerator("your-api-key-here")
feedback_collector = FeedbackCollector()
iterator = InteractiveCreativeIterator(creative_generator, feedback_collector)

initial_idea = "A smart mirror that provides personalized health and fitness recommendations"
final_idea, history = iterator.iterate(initial_idea)

print(f"\nFinal idea: {final_idea}")
print("\nIteration history:")
for item in history:
    print(f"Iteration {item['iteration']}:")
    print(f"Original: {item['original_idea']}")
    print(f"Feedback: {item['feedback']}")
    print(f"Improved: {item['improved_idea']}\n")
```

### 9.5.3 创意组合与融合

实现一个系统，将人类提出的创意元素与AI生成的创意进行智能组合和融合。

示例（创意组合与融合系统）：

```python
import random

class CreativeFusionSystem:
    def __init__(self, creative_generator):
        self.generator = creative_generator
        self.human_elements = []

    def add_human_element(self, element):
        self.human_elements.append(element)

    def generate_fusion(self, num_elements=2):
        if len(self.human_elements) < num_elements:
            raise ValueError("Not enough human elements for fusion")
        
        selected_elements = random.sample(self.human_elements, num_elements)
        fusion_prompt = f"Create a new idea by combining these elements: {', '.join(selected_elements)}\n\nFused idea:"
        
        fused_idea = self.generator.generate(fusion_prompt)
        return fused_idea, selected_elements

# 使用示例
fusion_system = CreativeFusionSystem(creative_generator)

# 添加人类提供的创意元素
fusion_system.add_human_element("eco-friendly materials")
fusion_system.add_human_element("gamification")
fusion_system.add_human_element("voice recognition")
fusion_system.add_human_element("augmented reality")

# 生成融合创意
fused_idea, used_elements = fusion_system.generate_fusion()
print(f"Fused idea: {fused_idea}")
print(f"Used elements: {used_elements}")
```

## 9.6 创意展示与应用

创意的有效展示和应用对于将概念转化为实际价值至关重要。

### 9.6.1 多模态创意呈现

开发一个系统，能够以多种形式（文本、图像、3D模型等）呈现生成的创意。

示例（多模态创意呈现系统）：

```python
from PIL import Image
import requests
from io import BytesIO

class MultimodalPresentationSystem:
    def __init__(self, text_generator, image_generator):
        self.text_generator = text_generator
        self.image_generator = image_generator

    def generate_text_description(self, idea):
        prompt = f"Provide a detailed description of the following idea:\n\n{idea}\n\nDescription:"
        return self.text_generator.generate(prompt)

    def generate_image(self, idea):
        prompt = f"Create an image representing the idea: {idea}"
        image_url = self.image_generator.generate(prompt)
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content))

    def present_idea(self, idea):
        description = self.generate_text_description(idea)
        image = self.generate_image(idea)
        
        print("Idea:", idea)
        print("\nDescription:", description)
        image.show()

# 使用示例（假设我们有文本和图像生成器）
text_gen = ConditionalCreativeGenerator("your-text-api-key")
image_gen = ImageGenerator("your-image-api-key")  # 假设的图像生成器

presentation_system = MultimodalPresentationSystem(text_gen, image_gen)

idea = "A self-cleaning, energy-efficient smart window that adapts its tint based on sunlight and user preferences"
presentation_system.present_idea(idea)
```

### 9.6.2 创意原型快速生成

实现一个系统，能够为生成的创意快速创建概念原型或模拟演示。

示例（简单的创意原型生成器）：

```python
import matplotlib.pyplot as plt
import networkx as nx

class RapidPrototypeGenerator:
    def __init__(self):
        self.components = {}

    def add_component(self, name, description):
        self.components[name] = description

    def generate_system_diagram(self, idea):
        G = nx.Graph()
        
        # 假设我们能够从创意描述中提取关键组件
        key_components = self.extract_key_components(idea)
        
        for component in key_components:
            G.add_node(component)
        
        # 添加一些假设的连接
        for i in range(len(key_components) - 1):
            G.add_edge(key_components[i], key_components[i+1])
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=3000, font_size=10, font_weight='bold')
        
        plt.title(f"System Diagram: {idea}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def extract_key_components(self, idea):
        # 这里应该实现一个更复杂的组件提取逻辑
        # 简单起见，我们只是分割字符串
        return [word.capitalize() for word in idea.split() if len(word) > 3][:5]

# 使用示例
prototype_gen = RapidPrototypeGenerator()

idea = "A smart hydroponic system with AI-driven nutrient management and climate control"
prototype_gen.generate_system_diagram(idea)
```

### 9.6.3 版权保护与管理

实现一个系统来管理和保护生成的创意，包括版权登记和相似度检查。

示例（简单的创意版权管理系统）：

```python
import hashlib
import datetime

class CreativeCopyrightManager:
    def __init__(self):
        self.copyright_registry = {}

    def register_copyright(self, idea, author):
        idea_hash = self.generate_hash(idea)
        timestamp = datetime.datetime.now().isoformat()
        
        if idea_hash in self.copyright_registry:
            return False, "This idea or a very similar one has already been registered."
        
        self.copyright_registry[idea_hash] = {
            "idea": idea,
            "author": author,
            "timestamp": timestamp
        }
        
        return True, f"Copyright registered. Registration ID: {idea_hash}"

    def check_similarity(self, new_idea):
        new_idea_hash = self.generate_hash(new_idea)
        
        for registered_hash, info in self.copyright_registry.items():
            if self.calculate_similarity(new_idea_hash, registered_hash) > 0.8:
                return True, info
        
        return False, None

    def generate_hash(self, idea):
        return hashlib.sha256(idea.encode()).hexdigest()

    def calculate_similarity(self, hash1, hash2):
        # 这是一个非常简化的相似度计算
        # 实际应用中应该使用更复杂的算法
        return sum(a == b for a, b in zip(hash1, hash2)) / len(hash1)

# 使用示例
copyright_manager = CreativeCopyrightManager()

# 注册一个创意
idea1 = "A smart watch that monitors mental health through voice analysis and suggests relaxation techniques"
success, message = copyright_manager.register_copyright(idea1, "Inventor A")
print(message)

# 尝试注册一个相似的创意
idea2 = "A wearable device that analyzes speech patterns to assess mental wellbeing and recommend stress-relief activities"
is_similar, existing_info = copyright_manager.check_similarity(idea2)

if is_similar:
    print(f"Warning: This idea is similar to an existing one registered by {existing_info['author']} on {existing_info['timestamp']}")
else:
    success, message = copyright_manager.register_copyright(idea2, "Inventor B")
    print(message)
```

这些组件和系统共同工作，可以创建一个全面的创意生成、优化、展示和管理平台。在实际应用中，这些系统可能需要更复杂的实现，包括更先进的自然语言处理、计算机视觉技术，以及更健壮的版权管理和相似度检测算法。此外，还需要考虑用户界面设计、数据安全性、可扩展性等方面，以构建一个真正实用和有效的创意生成 Agent 系统。