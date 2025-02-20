# import
import pickle
from collections import deque
import os
import matplotlib.pyplot as plt

from net import *
from config import *

# history load method
def load_history(file):
    '''
    file 경로에 저장되어 있는 history를 불러오는 함수
    '''
    try:
        with open(file, 'rb') as f:
            history = pickle.load(f)
        print(f"    Load history from \'{file}\'")

    except FileNotFoundError:
        history = deque(maxlen = MEM_SIZE) # 파일이 비어있는 경우 빈 리스트 생성
        print(f"    Create \'{file}\'")

    return history

# save data method
def save_data(file, data):
    '''
    file 경로에 history를 저장하는 함수
    '''
    with open(file, 'wb') as f:
        pickle.dump(data, f)
    print(f"    Save \'{file}\'")


# model 파라미터를 가져오는 함수
def load_model(file, model):
    '''
    file 경로에 저장되어 있는 모델 파라미터로 모델을 불러오는 함수
    '''
    model = model
    try:
        with open(file, 'rb') as f:
            model.load_state_dict(pickle.load(f))
        print(f"    Load model from \'{file}\'")
    except FileNotFoundError:
        print(f"    Create \'{file}\'")
    return model


# 최근 모델 저장 함수
def save_model(file, model):
    '''
    file 경로에 모델 파라미터를 저장하는 함수
    '''
    with open(file, 'wb') as f:
        pickle.dump(model.state_dict(), f)
    print(f"    Save model parameters in \'{file}\'")

