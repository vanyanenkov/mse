import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.core.hyperparameter_search import grid_search_params, random_search_params

def test_grid_search_params_base_case():
    assert grid_search_params({"lr0": [0.01, 0.00001], "test" : ["test1", "test2"]}) == [{'lr0': 0.01, 'test': 'test1'}, {'lr0': 0.01, 'test': 'test2'}, {'lr0': 1e-05, 'test': 'test1'}, {'lr0': 1e-05, 'test': 'test2'}]
    
def test_grid_search_params_empty_list():   
    assert grid_search_params({"empty_param": [], "lr": [1, 2]}) == [{'lr': 1}, {'lr': 2}]

def test_grid_search_params_fixed_parameter():
    assert grid_search_params({"lr0": [1], "str": ["s1", "s2"]}) == [{'lr0': 1, 'str': 's1'}, {'lr0': 1, 'str': 's2'}]

def test_grid_search_params_full_empty():
    assert grid_search_params({"lr0": [], "test": []}) == [{}]

def test_random_search_params_base_case():
    generated_params = random_search_params({"lr0": [1, 3], "test": ["test1", "test2"]}, 2)
    answer = False
    if len(generated_params) == 2:
        answer = True

        for dict in generated_params:
            if len(dict) != 2:
                answer = False

    assert answer == True 

def test_random_search_params_empty_list():
    generated_params = random_search_params({"lr0": [], "test": ["test1", "test2"]}, 2)

    flag = False
    if generated_params == [{'test': 'test2'}, {'test': 'test1'}] or generated_params == [{'test': 'test1'}, {'test': 'test2'}]:
        flag = True
    
    assert flag == True

def test_random_search_params_invalid_combination_count():
    generated_params = random_search_params({"epochs": [1, 2], "test": ["test1", "test2"]}, 10)
    
    flag = False
    true_params = [{'epochs': 2, 'test': 'test2'}, {'epochs': 2, 'test': 'test1'}, {'epochs': 1, 'test': 'test1'}, {'epochs': 1, 'test': 'test2'}]
    if len(generated_params) == 4:
        flag = True
        for combination in generated_params:
            if combination not in true_params:
                flag = False

    assert flag == True

def test_random_search_params_full_empty():

    generated_params = random_search_params({"epochs": [], "test": []}, 2)
    assert generated_params == [{}]