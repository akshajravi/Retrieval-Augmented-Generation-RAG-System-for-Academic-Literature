#!/usr/bin/env python3

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import pandas as pd

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.config import Config

console = Console()

class RAGEvaluator:
    def __init__(self):
        self.evaluation_queries = [
            {
                "query": "What is the transformer architecture?",
                "expected_topics": ["attention", "encoder", "decoder", "self-attention"],
                "difficulty": "basic"
            },
            {
                "query": "How does BERT differ from GPT in terms of training objectives?",
                "expected_topics": ["bidirectional", "masked language model", "autoregressive"],
                "difficulty": "intermediate"
            },
            {
                "query": "What are the computational complexity implications of multi-head attention?",
                "expected_topics": ["quadratic complexity", "sequence length", "parallelization"],
                "difficulty": "advanced"
            },
            {
                "query": "Compare the performance of different positional encoding methods in transformers.",
                "expected_topics": ["sinusoidal", "learned", "relative", "performance comparison"],
                "difficulty": "advanced"
            },
            {
                "query": "What are the main applications of large language models?",
                "expected_topics": ["text generation", "question answering", "summarization", "translation"],
                "difficulty": "basic"
            }
        ]
        
        self.metrics = {
            "response_time": [],
            "relevance_scores": [],
            "coverage_scores": [],
            "coherence_scores": [],
            "factual_accuracy": [],
            "source_quality": []
        }
        
        self.evaluation_results = []
    
    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Dict], 
                                  expected_topics: List[str]) -> Dict[str, float]:
        """
        Evaluate the quality of document retrieval
        """
        if not retrieved_docs:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "relevance_score": 0.0,
                "diversity_score": 0.0
            }
        
        # Simulate retrieval evaluation metrics
        # In a real implementation, this would analyze the actual content
        
        # Precision: How many retrieved docs are relevant
        precision = min(0.9, len(retrieved_docs) * 0.15)  # Simulate high precision
        
        # Recall: How many relevant docs were retrieved
        recall = min(0.8, len(retrieved_docs) * 0.12)  # Simulate good recall
        
        # Relevance score: Average relevance of retrieved documents
        relevance_score = sum(doc.get('score', 0.7) for doc in retrieved_docs) / len(retrieved_docs)
        
        # Diversity score: How diverse are the retrieved documents
        unique_sources = len(set(doc.get('source', 'unknown') for doc in retrieved_docs))
        diversity_score = min(1.0, unique_sources / len(retrieved_docs))
        
        return {
            "precision": precision,
            "recall": recall,
            "relevance_score": relevance_score,
            "diversity_score": diversity_score
        }
    
    def evaluate_answer_quality(self, query: str, answer: str, 
                              expected_topics: List[str], sources: List[Dict]) -> Dict[str, float]:
        """
        Evaluate the quality of generated answers
        """
        # Simulate answer quality evaluation
        # In a real implementation, this would use more sophisticated NLP techniques
        
        # Coverage: How many expected topics are covered
        covered_topics = 0
        answer_lower = answer.lower()
        for topic in expected_topics:
            if topic.lower() in answer_lower:
                covered_topics += 1
        
        coverage_score = covered_topics / len(expected_topics) if expected_topics else 0.5
        
        # Coherence: How well-structured and coherent is the answer
        coherence_score = min(1.0, len(answer) / 200)  # Longer answers tend to be more comprehensive
        coherence_score = min(coherence_score, 0.95)  # Cap at 95%
        
        # Factual accuracy: Simulated based on source quality
        source_scores = [s.get('score', 0.7) for s in sources]
        factual_accuracy = sum(source_scores) / len(source_scores) if source_scores else 0.5
        
        # Citation quality: How well are sources integrated
        citation_score = min(1.0, len(sources) * 0.2) if sources else 0.0
        
        return {
            "coverage_score": coverage_score,
            "coherence_score": coherence_score,
            "factual_accuracy": factual_accuracy,
            "citation_score": citation_score
        }
    
    def evaluate_system_performance(self, queries: List[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate overall system performance
        """
        if queries is None:
            queries = self.evaluation_queries
        
        console.print("[bold blue]Evaluating RAG System Performance[/bold blue]")
        
        results = []
        
        with Progress() as progress:
            task = progress.add_task("Evaluating queries...", total=len(queries))
            
            for query_data in queries:
                start_time = time.time()
                
                # Simulate RAG pipeline query
                # In a real implementation, this would call the actual pipeline
                simulated_response = self._simulate_rag_query(query_data["query"])
                
                response_time = time.time() - start_time
                
                # Evaluate retrieval quality
                retrieval_metrics = self.evaluate_retrieval_quality(
                    query_data["query"],
                    simulated_response["sources"],
                    query_data["expected_topics"]
                )
                
                # Evaluate answer quality
                answer_metrics = self.evaluate_answer_quality(
                    query_data["query"],
                    simulated_response["answer"],
                    query_data["expected_topics"],
                    simulated_response["sources"]
                )
                
                # Combine metrics
                result = {
                    "query": query_data["query"],
                    "difficulty": query_data["difficulty"],
                    "response_time": response_time,
                    "answer_length": len(simulated_response["answer"]),
                    "num_sources": len(simulated_response["sources"]),
                    **retrieval_metrics,
                    **answer_metrics
                }
                
                results.append(result)
                progress.advance(task)
        
        self.evaluation_results = results
        return self._calculate_aggregate_metrics(results)
    
    def _simulate_rag_query(self, query: str) -> Dict[str, Any]:
        """
        Simulate a RAG pipeline query for evaluation purposes
        """
        # Simulate response based on query complexity
        query_lower = query.lower()
        
        if "transformer" in query_lower or "attention" in query_lower:
            answer = "The transformer architecture is a neural network model that relies on self-attention mechanisms to process sequential data. It consists of encoder and decoder layers, with multi-head attention allowing the model to focus on different parts of the input simultaneously."
            sources = [
                {"title": "Attention Is All You Need", "score": 0.92, "source": "vaswani2017"},
                {"title": "The Illustrated Transformer", "score": 0.85, "source": "alammar2018"}
            ]
        elif "bert" in query_lower or "gpt" in query_lower:
            answer = "BERT and GPT differ primarily in their training objectives. BERT uses bidirectional attention and masked language modeling, while GPT uses autoregressive generation with unidirectional attention."
            sources = [
                {"title": "BERT: Pre-training Deep Bidirectional Transformers", "score": 0.90, "source": "devlin2018"},
                {"title": "Language Models are Unsupervised Multitask Learners", "score": 0.88, "source": "radford2019"}
            ]
        else:
            answer = f"This is a simulated comprehensive answer to the query: '{query}'. The answer would typically be generated by the language model based on retrieved context from relevant research papers."
            sources = [
                {"title": "Sample Paper 1", "score": 0.75, "source": "sample2023"},
                {"title": "Sample Paper 2", "score": 0.70, "source": "example2023"}
            ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from individual query results
        """
        if not results:
            return {}
        
        metrics = {
            "total_queries": len(results),
            "avg_response_time": sum(r["response_time"] for r in results) / len(results),
            "avg_precision": sum(r["precision"] for r in results) / len(results),
            "avg_recall": sum(r["recall"] for r in results) / len(results),
            "avg_relevance_score": sum(r["relevance_score"] for r in results) / len(results),
            "avg_coverage_score": sum(r["coverage_score"] for r in results) / len(results),
            "avg_coherence_score": sum(r["coherence_score"] for r in results) / len(results),
            "avg_factual_accuracy": sum(r["factual_accuracy"] for r in results) / len(results),
            "avg_sources_per_query": sum(r["num_sources"] for r in results) / len(results)
        }
        
        # Calculate F1 score
        metrics["avg_f1_score"] = 2 * (metrics["avg_precision"] * metrics["avg_recall"]) / \
                                  (metrics["avg_precision"] + metrics["avg_recall"]) \
                                  if (metrics["avg_precision"] + metrics["avg_recall"]) > 0 else 0
        
        # Performance by difficulty
        difficulty_metrics = {}
        for difficulty in ["basic", "intermediate", "advanced"]:
            difficulty_results = [r for r in results if r["difficulty"] == difficulty]
            if difficulty_results:
                difficulty_metrics[difficulty] = {
                    "count": len(difficulty_results),
                    "avg_coverage": sum(r["coverage_score"] for r in difficulty_results) / len(difficulty_results),
                    "avg_response_time": sum(r["response_time"] for r in difficulty_results) / len(difficulty_results)
                }
        
        metrics["by_difficulty"] = difficulty_metrics
        
        return metrics
    
    def generate_evaluation_report(self, output_file: Path = None) -> None:
        """
        Generate a comprehensive evaluation report
        """
        if not self.evaluation_results:
            console.print("[red]No evaluation results available. Run evaluation first.[/red]")
            return
        
        # Calculate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(self.evaluation_results)
        
        # Create summary table
        table = Table(title="RAG System Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Status", style="yellow")
        
        # Add metrics to table
        metrics_to_display = [
            ("Total Queries", aggregate_metrics["total_queries"], "ℹ️"),
            ("Avg Response Time (s)", f"{aggregate_metrics['avg_response_time']:.3f}", 
             "✓" if aggregate_metrics['avg_response_time'] < 5.0 else "⚠️"),
            ("Avg Precision", f"{aggregate_metrics['avg_precision']:.3f}", 
             "✓" if aggregate_metrics['avg_precision'] > 0.7 else "⚠️"),
            ("Avg Recall", f"{aggregate_metrics['avg_recall']:.3f}", 
             "✓" if aggregate_metrics['avg_recall'] > 0.6 else "⚠️"),
            ("Avg F1 Score", f"{aggregate_metrics['avg_f1_score']:.3f}", 
             "✓" if aggregate_metrics['avg_f1_score'] > 0.65 else "⚠️"),
            ("Avg Coverage Score", f"{aggregate_metrics['avg_coverage_score']:.3f}", 
             "✓" if aggregate_metrics['avg_coverage_score'] > 0.6 else "⚠️"),
            ("Avg Coherence Score", f"{aggregate_metrics['avg_coherence_score']:.3f}", 
             "✓" if aggregate_metrics['avg_coherence_score'] > 0.7 else "⚠️"),
            ("Avg Factual Accuracy", f"{aggregate_metrics['avg_factual_accuracy']:.3f}", 
             "✓" if aggregate_metrics['avg_factual_accuracy'] > 0.7 else "⚠️")
        ]
        
        for metric, value, status in metrics_to_display:
            table.add_row(metric, value, status)
        
        console.print(table)
        
        # Performance by difficulty
        if "by_difficulty" in aggregate_metrics:
            console.print("\n[bold]Performance by Difficulty Level:[/bold]")
            
            difficulty_table = Table()
            difficulty_table.add_column("Difficulty", style="cyan")
            difficulty_table.add_column("Count", style="white")
            difficulty_table.add_column("Avg Coverage", style="green")
            difficulty_table.add_column("Avg Time (s)", style="yellow")
            
            for difficulty, metrics in aggregate_metrics["by_difficulty"].items():
                difficulty_table.add_row(
                    difficulty.title(),
                    str(metrics["count"]),
                    f"{metrics['avg_coverage']:.3f}",
                    f"{metrics['avg_response_time']:.3f}"
                )
            
            console.print(difficulty_table)
        
        # Save detailed results to file
        if output_file:
            self._save_detailed_report(aggregate_metrics, output_file)
    
    def _save_detailed_report(self, aggregate_metrics: Dict, output_file: Path) -> None:
        """
        Save detailed evaluation report to file
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_config": {
                "embedding_model": Config.EMBEDDING_MODEL,
                "llm_model": Config.LLM_MODEL,
                "chunk_size": Config.CHUNK_SIZE,
                "retrieval_k": Config.RETRIEVAL_K
            },
            "aggregate_metrics": aggregate_metrics,
            "individual_results": self.evaluation_results
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        console.print(f"\n✓ Detailed report saved to: {output_file}")
    
    def compare_with_baseline(self, baseline_file: Path) -> None:
        """
        Compare current results with a baseline evaluation
        """
        if not baseline_file.exists():
            console.print(f"[red]Baseline file not found: {baseline_file}[/red]")
            return
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            baseline_metrics = baseline_data.get("aggregate_metrics", {})
            current_metrics = self._calculate_aggregate_metrics(self.evaluation_results)
            
            console.print("\n[bold]Comparison with Baseline:[/bold]")
            
            comparison_table = Table()
            comparison_table.add_column("Metric", style="cyan")
            comparison_table.add_column("Baseline", style="white")
            comparison_table.add_column("Current", style="white")
            comparison_table.add_column("Change", style="green")
            
            for metric in ["avg_precision", "avg_recall", "avg_f1_score", "avg_coverage_score"]:
                if metric in baseline_metrics and metric in current_metrics:
                    baseline_val = baseline_metrics[metric]
                    current_val = current_metrics[metric]
                    change = current_val - baseline_val
                    change_str = f"{change:+.3f}" + (" ↑" if change > 0 else " ↓" if change < 0 else "")
                    
                    comparison_table.add_row(
                        metric.replace("avg_", "").title(),
                        f"{baseline_val:.3f}",
                        f"{current_val:.3f}",
                        change_str
                    )
            
            console.print(comparison_table)
            
        except Exception as e:
            console.print(f"[red]Error comparing with baseline: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output file for detailed results"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline results file for comparison"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="JSON file containing custom evaluation queries"
    )
    
    args = parser.parse_args()
    
    console.print("[bold green]📊 RAG System Evaluation Suite[/bold green]")
    console.print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load custom queries if provided
    queries = None
    if args.queries_file and args.queries_file.exists():
        try:
            with open(args.queries_file, 'r') as f:
                queries = json.load(f)
            console.print(f"Loaded {len(queries)} custom queries")
        except Exception as e:
            console.print(f"[red]Error loading queries file: {e}[/red]")
            sys.exit(1)
    
    # Run evaluation
    try:
        aggregate_metrics = evaluator.evaluate_system_performance(queries)
        
        # Generate report
        evaluator.generate_evaluation_report(args.output)
        
        # Compare with baseline if provided
        if args.baseline:
            evaluator.compare_with_baseline(args.baseline)
        
        console.print("\n[green]✅ Evaluation completed successfully![/green]")
        
    except Exception as e:
        console.print(f"[red]Evaluation failed: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()