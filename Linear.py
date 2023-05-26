import numpy as np

class Linear():
    ## 객체 생성자
    def __init__(self, learning_rate = 0.001, epoch = 1):
        self.learning_rate = learning_rate
        self.epoch = epoch
        
    ## 훈련 함수    
    def fit(self, input_array, target_array):
        a = 0
        b = 0
        n = input_array.shape[0]
        if type(input_array) != np.ndarray or type(target_array) != np.ndarray:
            print("input 이나 target이 array가 아닙니다.(데이터의 타입을 확인해 주세요)")
            return
        
        else:
            ## 데이터의 숫자 * 횟수 만큼 반복
            for _ in range(self.epoch * n):
                ## 계산된 계수와 절편 확인
                print('a =', a, 'b=', b)
                
                ## 계수와 절편을 계산하기 위한 편미분 값
                delta_a = -2 * np.sum(input_array * (target_array - ((a * input_array) + b))) / len(input_array)
                delta_b = -2 * np.sum(target_array - ((a * input_array) + b)) / len(input_array)
                
                ## 계수와 절편 계산
                a = a - self.learning_rate * delta_a
                b = b - self.learning_rate * delta_b
            
            ## 계수와 절편 저장    
            self.coef = a
            self.intercept = b
    
    ## 저장된 계수와 절편 값으로 선형 방정식을 만들어 데이터값을 예측 해주는 함수            
    def predict(self, input_array):
        return (self.coef * input_array + self.intercept)
    
    ## 기존의 선형방정식에서 추가로 학습해 주는 함수
    def partial_fit(self, input_array, target_array):
        n = input_array.shape[0]
        if type(input_array) != np.ndarray or type(target_array) != np.ndarray:
            print("input 이나 target이 array가 아닙니다.(데이터의 타입을 확인해 주세요)")
            return
        
        else:
            ## 기존에 저장되있는 계수, 절편, 오차값을 출력
            print('a =', self.coef, 'b=', self.intercept, 'loss=', (np.sum((target_array - self.predict(input_array)) ** 2) / n))
            
            ## 데이터의 숫자 * 횟수 만큼 반복
            for _ in range(self.epoch * n):
                ## 계수와 절편을 계산하기 위한 편미분 값
                delta_a = -2 * np.sum(input_array * (target_array - ((self.coef * input_array) + self.intercept))) / len(input_array)
                delta_b = -2 * np.sum(target_array - ((self.coef * input_array) + self.intercept)) / len(input_array)
                
                ## 계수와 절편 계산
                self.coef = self.coef - self.learning_rate * delta_a
                self.intercept = self.intercept - self.learning_rate * delta_b
                
                ## 계수, 절편, 오차값을 출력
                print('a =', self.coef, 'b=', self.intercept, 'loss=', (np.sum((target_array - self.predict(input_array)) ** 2) / n))