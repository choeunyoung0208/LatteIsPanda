# https://kkamikoon.tistory.com/135
# https://jbmpa.com/pygame/10
# 반복문 안에 있어야 소리가 재생 됨

import pygame # 효과음 재생하기 위해 import
import time # 그냥 테스트용
i=0 # 그냥 테스트용

pygame.init()       # 음악 재생하기 위해 필요한 요소 초기화
pygame.mixer.init() # 음악 재생하기 위해 필요한 요소 초기화
Sound = pygame.mixer.Sound('test2.mp3') # 불러올 mp3 파일 설정 및 인스턴스 생성
Sound.set_volume(0.1)   # 0.0 ~ 1.0 볼륨 조절

if (pygame.mixer.get_busy() == False):  # 효과음이 재생되고있지 않을 경우
    Sound.play()  # 효과음 재생
    print("효과음 재생")
    # play(loops=0, maxtime=0, fade_ms=0)

# 아래는 테스트 해보는 것
while(True):
    print("hello")
    time.sleep(0.1)
    i=i+1
    if(i==20) :
        Sound.stop() # 멈추는지 테스트용
        
