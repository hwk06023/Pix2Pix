# Pix2Pix

2019 선린인터넷 고등학교 이모션 시연회 때 발표한 프로젝트입니다.

<img src=https://github.com/hwk06023/Pix2Pix/blob/master/images/main.png width=500 heigth=500>

Pix2Pix를 활용하여 건물의 앞면을 그리면 실제 사진처럼 변환해주는 프로젝트를 진행했습니다. <br/>
웹을 활용하여 많은 사람들이 직접 접속해서 참여해볼 수 있도록 진행했습니다.

## Web







## Model

### DCGAN(Deep Convolutional Generative Adversarial Networks)

### U-Net
<img src=https://github.com/hwk06023/Pix2Pix/blob/master/images/u-net.png width=500 heigth=500>
U-Net: Convolutional Networks for Biomedical Image Segmentation 라는 논문에서 소개되었습니다. <br/>

U-Net은 다른 네트워크들과 다르게 줄여 부르는게 아닌, 네트워크의 구조가 U의 모양이라 U-Net으로 불리기 시작했습니다. <br/>

Pix2Pix에서 U-Net을 사용해서 Conv연산과 UpConv연산을 거치며 흐려지는 문제를 개선했습니다. <br/> <br/>




#### 한줄 요약 : DCGAN의 구조에서 Generater 부분에 U-Net 구조를 사용하였다.



## 고동현우.한국

[고동현우.한국](고동현우.한국) 으로 접속할 수 있지만.. <br/>
서버 켜두면 돈 나가서 평소엔 서버를 닫고 있습니다 T^T

### 참조
William Falcon님의 문서를 참조했습니다.
