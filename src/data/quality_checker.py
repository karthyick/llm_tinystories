"""Data Quality Checker for training datasets.

This module provides tools to validate dataset quality before training:
- Detects artifacts (HTML tags, URLs, special tokens)
- Checks for malformed text
- Validates text statistics
- Reports quality issues

Prevents training on corrupted or low-quality data.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Check dataset quality before training."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        sample_size: Optional[int] = 10000,
        strict: bool = False,
    ):
        """Initialize quality checker.

        Args:
            dataset_name: Name of dataset (e.g., "roneneldan/TinyStories")
            split: Dataset split to check ("train" or "validation")
            sample_size: Number of samples to check (None for all)
            strict: If True, raise errors on issues; if False, only warn
        """
        self.dataset_name = dataset_name
        self.split = split
        self.sample_size = sample_size
        self.strict = strict

        # Quality metrics
        self.issues: Dict[str, List[Tuple[int, str]]] = {
            "html_tags": [],
            "urls": [],
            "emails": [],
            "excessive_punctuation": [],
            "malformed_unicode": [],
            "empty_text": [],
            "extremely_short": [],
            "extremely_long": [],
            "suspicious_patterns": [],
            "special_tokens": [],
        }

        self.stats = {
            "total_samples": 0,
            "total_chars": 0,
            "total_words": 0,
            "avg_length": 0,
            "vocabulary_size": 0,
        }

    def check_quality(self) -> Dict:
        """Run all quality checks and return results.

        Returns:
            Dictionary with quality report and pass/fail status
        """
        logger.info(f"Loading dataset {self.dataset_name} ({self.split} split)...")

        # Load dataset
        if "tinystories" in self.dataset_name.lower():
            dataset = load_dataset("roneneldan/TinyStories", split=self.split)
        elif "wikitext" in self.dataset_name.lower():
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=self.split, trust_remote_code=True)
        else:
            dataset = load_dataset(self.dataset_name, split=self.split)

        # Limit sample size if requested
        if self.sample_size and len(dataset) > self.sample_size:
            logger.info(f"Sampling {self.sample_size} examples from {len(dataset)} total")
            indices = range(0, len(dataset), len(dataset) // self.sample_size)
            dataset = dataset.select(list(indices)[:self.sample_size])

        logger.info(f"Checking quality of {len(dataset)} examples...")

        # Run checks
        vocabulary = set()

        for idx, example in enumerate(tqdm(dataset, desc="Quality Check")):
            text = example.get("text", "")

            # Update stats
            self.stats["total_samples"] += 1
            self.stats["total_chars"] += len(text)
            words = text.split()
            self.stats["total_words"] += len(words)
            vocabulary.update(words)

            # Run individual checks
            self._check_html_tags(idx, text)
            self._check_urls(idx, text)
            self._check_emails(idx, text)
            self._check_excessive_punctuation(idx, text)
            self._check_malformed_unicode(idx, text)
            self._check_empty_text(idx, text)
            self._check_length_extremes(idx, text)
            self._check_suspicious_patterns(idx, text)
            self._check_special_tokens(idx, text)

        # Calculate final stats
        if self.stats["total_samples"] > 0:
            self.stats["avg_length"] = self.stats["total_chars"] / self.stats["total_samples"]
            self.stats["avg_words"] = self.stats["total_words"] / self.stats["total_samples"]
        self.stats["vocabulary_size"] = len(vocabulary)

        # Generate report
        report = self._generate_report()

        return report

    def _check_html_tags(self, idx: int, text: str):
        """Check for HTML tags."""
        html_pattern = r'<[^>]+>'
        if re.search(html_pattern, text):
            self.issues["html_tags"].append((idx, text[:100]))

    def _check_urls(self, idx: int, text: str):
        """Check for URLs."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        if re.search(url_pattern, text):
            self.issues["urls"].append((idx, text[:100]))

    def _check_emails(self, idx: int, text: str):
        """Check for email addresses."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        if re.search(email_pattern, text):
            self.issues["emails"].append((idx, text[:100]))

    def _check_excessive_punctuation(self, idx: int, text: str):
        """Check for excessive punctuation (possible artifacts)."""
        # More than 5 consecutive punctuation marks
        if re.search(r'[!?.,;:]{5,}', text):
            self.issues["excessive_punctuation"].append((idx, text[:100]))

        # More than 20% punctuation
        if len(text) > 0:
            punct_count = sum(1 for c in text if c in '!?.,;:')
            if punct_count / len(text) > 0.2:
                self.issues["excessive_punctuation"].append((idx, text[:100]))

    def _check_malformed_unicode(self, idx: int, text: str):
        """Check for malformed Unicode characters."""
        # Look for replacement characters or control characters
        if '�' in text or '\ufffd' in text:
            self.issues["malformed_unicode"].append((idx, text[:100]))

        # Control characters (excluding whitespace)
        if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', text):
            self.issues["malformed_unicode"].append((idx, text[:100]))

    def _check_empty_text(self, idx: int, text: str):
        """Check for empty or whitespace-only text."""
        if not text or not text.strip():
            self.issues["empty_text"].append((idx, text))

    def _check_length_extremes(self, idx: int, text: str):
        """Check for extremely short or long text."""
        if len(text.strip()) < 10:
            self.issues["extremely_short"].append((idx, text))
        elif len(text) > 50000:  # Suspiciously long
            self.issues["extremely_long"].append((idx, text[:100]))

    def _check_suspicious_patterns(self, idx: int, text: str):
        """Check for suspicious patterns."""
        # Repeated characters (e.g., "aaaaaa" more than 10 times)
        if re.search(r'(.)\1{10,}', text):
            self.issues["suspicious_patterns"].append((idx, text[:100]))

        # Excessive whitespace
        if re.search(r'\s{10,}', text):
            self.issues["suspicious_patterns"].append((idx, text[:100]))

    def _check_special_tokens(self, idx: int, text: str):
        """Check for special tokens that shouldn't be in raw text."""
        # Common tokenizer special tokens
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<|endoftext|>', '<pad>', '<unk>']
        for token in special_tokens:
            if token in text:
                self.issues["special_tokens"].append((idx, text[:100]))
                break

    def _generate_report(self) -> Dict:
        """Generate quality report.

        Returns:
            Dictionary with quality metrics and pass/fail status
        """
        total_issues = sum(len(issues) for issues in self.issues.values())
        issue_percentage = (total_issues / self.stats["total_samples"] * 100) if self.stats["total_samples"] > 0 else 0

        # Determine quality level
        if issue_percentage == 0:
            quality_level = "EXCELLENT"
            passed = True
        elif issue_percentage < 1:
            quality_level = "GOOD"
            passed = True
        elif issue_percentage < 5:
            quality_level = "ACCEPTABLE"
            passed = not self.strict
        elif issue_percentage < 10:
            quality_level = "POOR"
            passed = False
        else:
            quality_level = "CRITICAL"
            passed = False

        report = {
            "dataset": self.dataset_name,
            "split": self.split,
            "quality_level": quality_level,
            "passed": passed,
            "stats": self.stats,
            "issues": {
                key: {
                    "count": len(value),
                    "percentage": (len(value) / self.stats["total_samples"] * 100) if self.stats["total_samples"] > 0 else 0,
                    "samples": value[:3]  # First 3 examples
                }
                for key, value in self.issues.items() if len(value) > 0
            },
            "total_issues": total_issues,
            "issue_percentage": issue_percentage,
        }

        return report

    def print_report(self, report: Dict):
        """Print formatted quality report.

        Args:
            report: Report dictionary from check_quality()
        """
        logger.info("\n" + "=" * 70)
        logger.info("DATA QUALITY REPORT")
        logger.info("=" * 70)
        logger.info(f"Dataset: {report['dataset']} ({report['split']} split)")
        logger.info(f"Quality Level: {report['quality_level']}")
        logger.info(f"Status: {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
        logger.info("")

        # Statistics
        logger.info("Statistics:")
        logger.info(f"  Total Samples: {report['stats']['total_samples']:,}")
        logger.info(f"  Avg Length: {report['stats']['avg_length']:.1f} chars")
        logger.info(f"  Avg Words: {report['stats'].get('avg_words', 0):.1f} words")
        logger.info(f"  Vocabulary Size: {report['stats']['vocabulary_size']:,}")
        logger.info("")

        # Issues
        if report['issues']:
            logger.warning(f"Found {report['total_issues']} issues ({report['issue_percentage']:.2f}% of samples)")
            logger.warning("")
            for issue_type, details in report['issues'].items():
                logger.warning(f"  {issue_type.replace('_', ' ').title()}:")
                logger.warning(f"    Count: {details['count']} ({details['percentage']:.2f}%)")
                if details['samples']:
                    logger.warning(f"    Example: {details['samples'][0][1][:80]}...")
                logger.warning("")
        else:
            logger.info("✅ No quality issues found!")

        logger.info("=" * 70)

        # Recommendations
        if not report['passed']:
            logger.error("\n⚠️  DATA HAS QUALITY ISSUES - Training not recommended!")
            logger.error("Recommendations:")
            if report['issues'].get('html_tags'):
                logger.error("  - Remove HTML tags from text")
            if report['issues'].get('urls'):
                logger.error("  - Remove or mask URLs")
            if report['issues'].get('malformed_unicode'):
                logger.error("  - Fix Unicode encoding issues")
            if report['issues'].get('empty_text'):
                logger.error("  - Remove empty samples")
            logger.error("")


def check_dataset_quality(
    dataset_name: str,
    split: str = "train",
    sample_size: Optional[int] = 10000,
    strict: bool = False,
) -> bool:
    """Quick function to check dataset quality.

    Args:
        dataset_name: Dataset name or HuggingFace ID
        split: Split to check
        sample_size: Number of samples to check (None for all)
        strict: If True, fail on any issues

    Returns:
        True if quality is acceptable, False otherwise
    """
    checker = DataQualityChecker(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size,
        strict=strict,
    )

    report = checker.check_quality()
    checker.print_report(report)

    return report["passed"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check dataset quality")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of samples to check")
    parser.add_argument("--strict", action="store_true", help="Fail on any issues")

    args = parser.parse_args()

    passed = check_dataset_quality(
        dataset_name=args.dataset,
        split=args.split,
        sample_size=args.sample_size,
        strict=args.strict,
    )

    exit(0 if passed else 1)
