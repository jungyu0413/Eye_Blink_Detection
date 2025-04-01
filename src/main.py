import cv2
import numpy as np
import torchvision.transforms as transforms
import clean_realtime_module 
from clean_module import load_net
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning) 

# 모델 불러오기
net, reverse_index1, reverse_index2, max_len, cfg, device = load_net()

# 이미지 전처리 정의
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    transforms.Resize((cfg.input_size, cfg.input_size)),
    transforms.ToTensor(),
    normalize
])

color = (120, 175, 160)
ear_xy = []
bf_frame_bool = False
total_fps = []

# 웹캠으로부터 실시간 영상 받기
cap = cv2.VideoCapture(0)  # 0번 카메라는 기본 웹캠

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

mean_ear = []
each_fps = []
blink_initialized = False
cnt = 0
bf_frame = 0
bf_frame_ear = []
bl_bool = False

print("[INFO] 눈 깜빡임 인식 시작")

try:
    while True:
        ret, image = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        try:
            st_time = time.time()
            output_image, EAR, total_time, check, eye_region_landmarks = clean_realtime_module.demo_image(
                image, net, preprocess, cfg.input_size, cfg.net_stride, cfg.num_nb, cfg.use_gpu, device,
                reverse_index1, reverse_index2, max_len
            )

            for eye_cd in eye_region_landmarks:
                cv2.circle(output_image, eye_cd, 1, (255, 255, 255), 1)

            if not blink_initialized:
                mean_ear.append(EAR)
                txt = 'Initializing' + str('.'*(len(mean_ear)//6))
                if len(mean_ear) >= 30:
                    bf = np.mean(mean_ear)
                    blink_dn = bf * 0.7
                    blink_up = bf
                    blink_initialized = True
                    print(f"[INFO] EAR 평균 초기화 완료 - mean EAR: {bf:.4f}")
            else:
                # EAR에 따라 눈 깜빡임 탐지
                bf = np.mean(mean_ear[-30:])
                blink_dn = bf * 0.7
                blink_up = bf

                if EAR < blink_dn:
                    bl_bool = True

                if bl_bool and EAR > blink_up:
                    cnt += 1
                    bl_bool = False
                    bf_frame = 0
                    bf_frame_ear = []
                    for eye_cd in eye_region_landmarks:
                        cv2.circle(output_image, eye_cd, 1, (0, 255, 0), 2)

                elif bl_bool:
                    bf_frame += 1
                    bf_frame_ear.append(EAR)

            txt = f'Blink Count: {cnt}'
            cv2.putText(output_image, txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            ed_time = time.time()
            c_fps = 1 / (ed_time - st_time)
            total_fps.append(c_fps)

            cv2.imshow("Blink Detection - Live", output_image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC 키 누르면 종료
                break

        except Exception as e:
            print("예외 발생:", str(e))
            txt = 'Blink Fail'
            cv2.putText(image, txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("Blink Detection - Live", image)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

except KeyboardInterrupt:
    print("\n[INFO] 강제 종료 감지")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if total_fps:
        print('-----------------------------------')
        print(f'평균 FPS : {np.mean(total_fps):.2f}')
        print(f'최종 눈 깜빡임 횟수 : {cnt}')
