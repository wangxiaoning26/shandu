import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import asyncio
from shandu.research.researcher import ResearchResult, DeepResearcher


class TestResearchResult(unittest.TestCase):
    """Test the ResearchResult class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.timestamp = datetime(2023, 1, 1, 12, 0, 0)
        self.result = ResearchResult(
            query="test query",
            summary="This is a test summary.",
            sources=[{"url": "https://example.com", "title": "Example"}],
            subqueries=["subquery 1", "subquery 2"],
            depth=2,
            content_analysis=[{"subquery": "subquery 1", "analysis": "Analysis 1", "sources": ["https://example.com"]}],
            chain_of_thought=["thought 1", "thought 2"],
            research_stats={"elapsed_time_formatted": "10s", "sources_count": 1, "breadth": 3, "subqueries_count": 2},
            timestamp=self.timestamp
        )
    
    def test_initialization(self):
        """Test result initialization."""
        self.assertEqual(self.result.query, "test query")
        self.assertEqual(self.result.summary, "This is a test summary.")
        self.assertEqual(len(self.result.sources), 1)
        self.assertEqual(self.result.sources[0]["url"], "https://example.com")
        self.assertEqual(self.result.subqueries, ["subquery 1", "subquery 2"])
        self.assertEqual(self.result.depth, 2)
        self.assertEqual(self.result.timestamp, self.timestamp)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result_dict = self.result.to_dict()
        self.assertEqual(result_dict["query"], "test query")
        self.assertEqual(result_dict["summary"], "This is a test summary.")
        self.assertEqual(result_dict["sources"], [{"url": "https://example.com", "title": "Example"}])
        self.assertEqual(result_dict["subqueries"], ["subquery 1", "subquery 2"])
        self.assertEqual(result_dict["depth"], 2)
        self.assertEqual(result_dict["timestamp"], self.timestamp.isoformat())
    
    def test_to_markdown(self):
        """Test markdown conversion."""
        markdown = self.result.to_markdown()
        self.assertIsInstance(markdown, str)
        self.assertGreater(len(markdown), 0)
        # Check that key sections are included
        self.assertIn("# test query", markdown)
        self.assertIn("This is a test summary.", markdown)
        self.assertIn("## Research Process", markdown)
        self.assertIn("https://example.com", markdown)
    
    @patch('shandu.research.researcher.open')
    def test_save_to_file(self, mock_open):
        """Test saving to file."""
        # Mock open to prevent actual file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Test saving as markdown
        self.result.save_to_file("/test/path.md")
        mock_open.assert_called_with("/test/path.md", 'w', encoding='utf-8')
        
        # Test saving as JSON
        mock_open.reset_mock()
        self.result.save_to_file("/test/path.json")
        mock_open.assert_called_with("/test/path.json", 'w', encoding='utf-8')


class TestDeepResearcher(unittest.TestCase):
    """Test the DeepResearcher class."""
    
    @patch('shandu.research.researcher.os.makedirs')
    def test_initialization(self, mock_makedirs):
        """Test researcher initialization."""
        researcher = DeepResearcher(
            output_dir="/test/output",
            save_results=True,
            auto_save_interval=60
        )
        
        self.assertEqual(researcher.output_dir, "/test/output")
        self.assertEqual(researcher.save_results, True)
        self.assertEqual(researcher.auto_save_interval, 60)
        mock_makedirs.assert_called_once_with("/test/output", exist_ok=True)
    
    @patch('shandu.research.researcher.datetime')
    def test_get_output_path(self, mock_datetime):
        """Test output path generation."""
        mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)
        
        researcher = DeepResearcher(output_dir="/test/output", save_results=False)
        path = researcher.get_output_path("Test Query", "md")
        
        # Check that the path includes the query and timestamp
        self.assertTrue(path.startswith("/test/output/"))
        self.assertTrue("Test_Query" in path)
        self.assertTrue("20230101_120000" in path)
        self.assertTrue(path.endswith(".md"))
    
    @patch('shandu.research.researcher.ResearchGraph')
    @patch('shandu.research.researcher.ResearchAgent')
    @patch('asyncio.run')
    def test_research_sync(self, mock_asyncio_run, mock_agent, mock_graph):
        """Test synchronous research."""
        # Setup mocks
        mock_agent_instance = MagicMock()
        mock_graph_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        mock_graph.return_value = mock_graph_instance
        mock_result = MagicMock()
        mock_asyncio_run.return_value = mock_result
        
        # Call the method
        researcher = DeepResearcher(save_results=False)
        result = researcher.research_sync("test query", strategy="langgraph")
        
        # Verify the correct method was called
        mock_asyncio_run.assert_called_once()
        self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main()