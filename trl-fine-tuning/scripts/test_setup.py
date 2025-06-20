#!/usr/bin/env python3
"""
Test script to verify TRL GRPO setup
"""

import sys
import os
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
    ]
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            return False
    
    return True

def test_modules():
    """Test if our custom modules can be imported."""
    print("\n🔧 Testing custom modules...")
    
    modules = [
        ("config.grpo_config", "GRPO Config"),
        ("data.gsm8k_loader", "GSM8K Loader"),
        ("models.reward_function", "Reward Function"),
    ]
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            return False
    
    return True

def test_model_access():
    """Test if we can access Qwen models."""
    print("\n🤖 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        from config.grpo_config import get_model_candidates
        
        models_to_test = get_model_candidates()
        available_model = None
        
        for model_name in models_to_test:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print(f"✅ {model_name}: Available")
                available_model = model_name
                break
            except Exception as e:
                print(f"❌ {model_name}: Not available")
        
        return available_model is not None
        
    except Exception as e:
        print(f"❌ Model access test failed: {e}")
        return False

def test_dataset():
    """Test if we can access GSM8K dataset."""
    print("\n📊 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main")
        print(f"✅ GSM8K: {len(dataset['train'])} train, {len(dataset['test'])} test examples")
        return True
    except Exception as e:
        print(f"❌ GSM8K dataset: FAILED - {e}")
        return False

def test_reward_function():
    """Test the reward function."""
    print("\n🎯 Testing reward function...")
    
    try:
        from models.reward_function import GSM8KRewardFunction
        
        reward_fn = GSM8KRewardFunction()
        
        # Test cases
        test_cases = [
            ("The answer is 42", 42.0, True),
            ("#### 42", 42.0, True),
            ("The answer is 43", 42.0, False),
            ("No number", 42.0, False),
        ]
        
        for completion, ground_truth, expected in test_cases:
            rewards = reward_fn([completion], ground_truth=[ground_truth])
            is_correct = rewards[0] == 1.0
            status = "✅" if is_correct == expected else "❌"
            print(f"{status} '{completion[:15]}...' -> {rewards[0]}")
        
        return True
    except Exception as e:
        print(f"❌ Reward function: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TRL GRPO Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_modules),
        ("Model Access", test_model_access),
        ("Dataset Access", test_dataset),
        ("Reward Function", test_reward_function),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            results[test_name] = False
    
    print("\n" + "=" * 40)
    print("📋 Summary:")
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to train.")
        print("Run: bash scripts/train.sh")
    else:
        print("\n🚨 Some tests failed. Please fix issues above.")

if __name__ == "__main__":
    main() 