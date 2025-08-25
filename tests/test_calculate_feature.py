"""
Unit tests for the calculate feature in Agent S.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from gui_agents.s2_5.agents.grounding import OSWorldACI


class TestCalculateFeature(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create minimal engine parameters for testing
        engine_params_generation = {
            "engine_type": "openai",
            "model": "gpt-5"
        }
        
        engine_params_grounding = {
            "engine_type": "openai",
            "model": "gpt-5",
            "grounding_width": 1920,
            "grounding_height": 1080
        }
        
        self.agent = OSWorldACI(
            platform="darwin",
            engine_params_for_generation=engine_params_generation,
            engine_params_for_grounding=engine_params_grounding
        )
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic expression"""
        expression = "2 + 3 * 4"
        result = self.agent.calculate(expression)
        expected = """import math
import numpy as np
result = 2 + 3 * 4
print(result)
"""
        self.assertEqual(result, expected)
    
    def test_math_functions(self):
        """Test math functions expression"""
        expression = "math.sqrt(16)"
        result = self.agent.calculate(expression)
        expected = """import math
import numpy as np
result = math.sqrt(16)
print(result)
"""
        self.assertEqual(result, expected)
    
    def test_numpy_functions(self):
        """Test numpy functions expression"""
        expression = "np.sin(np.pi/2)"
        result = self.agent.calculate(expression)
        expected = """import math
import numpy as np
result = np.sin(np.pi/2)
print(result)
"""
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()