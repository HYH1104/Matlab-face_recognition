clc
clear all
close all
warning off
imgsize=[300,300];
[fn,pn,fi]=uigetfile('*.*','选择图片');%选择图片
rgb=imread([pn fn]);%读取原始图像
figure(1);
imshow(rgb);%彩色原始图像
title('人脸原始图像')
disp('原始图像大小:')
disp(size(rgb))
figure(2);
R=rgb(:,:,1);
G=rgb(:,:,2);
B=rgb(:,:,3);
subplot(131), imshow(R);title('R域强度图像')
subplot(132), imshow(G);title('G域强度图像')
subplot(133), imshow(B);title('B域强度图像')
maxR=im2double(max(max(R)));
maxG=im2double(max(max(G)));
maxB=im2double(max(max(B)));
k=(maxR+maxG+maxB)/3;
if k<0.9
rgb = imadjust(rgb,[0 0 0; k k k],[0 0 0;1 1 1],0.7);
end
figure(3)
imshow(rgb);
title('rgb强度处理图像')
figure(4)
myhsv=rgb2hsv(rgb);%对图像进行HSV处理
H=myhsv(:,:,1);
S=myhsv(:,:,2);
V=myhsv(:,:,3);
subplot(131), imshow(H);title('H域强度图像')
subplot(132), imshow(S);title('S域强度图像')
subplot(133), imshow(V);title('V域强度图像')
maxH=im2double(max(max(H)));
maxS=im2double(max(max(S)));
maxV=im2double(max(max(V)));
k=(maxH+maxS+maxV)/3;
if k<0.9
rgb = imadjust(rgb,[0 0 0; k k k],[0 0 0;1 1 1],0.7);
end
figure(5)
imshow(rgb);
title('hsv强度处理图像')
mFaceResult   = face_segment(rgb);
figure(6);
imshow(mFaceResult);%彩色原始图像
title('人脸部分图像')
disp('归一化人脸图像大小:')
disp(size(mFaceResult))
figure(7);
[cA1, cH1, cV1, cD1] = dwt2(mFaceResult, 'haar');%小波分量提取分量
HUtest(:,1)=Humoment(cA1).';
HUtest(:,2)=Humoment(cH1).';
HUtest(:,3)=Humoment(cV1).';
HUtest(:,4)=Humoment(cD1).';
subplot(221), imshow(cA1, []);title('近似系数')%显示近似系数
subplot(222), imshow(cH1, []);title('水平细节分量')%显示水平细节分量
subplot(223), imshow(cV1, []);title('垂直细节分量')%显示垂直细节分量
subplot(224), imshow(cD1, []);title('对角细节分量')%显示对角细节分量
figure(8)
A1=im2bw(cA1,0.5);
A2=im2bw(cH1,0.5);
A3=im2bw(cV1,0.5);
A4=im2bw(cD1,0.5);
subplot(221), imshow(A1, []);title('近似系数二值化图像')%显示近似系数
subplot(222), imshow(A2, []);title('水平细节分量二值化图像')%显示水平细节分量
subplot(223), imshow(A3, []);title('垂直细节分量二值化图像')%显示垂直细节分量
subplot(224), imshow(A4, []);title('对角细节分量二值化图像')%显示对角细节分量
figure(9);
img2=bwperim(A2);
img3=bwperim(A3);
img4=bwperim(A4);
subplot(221), imshow(A1, []);title('近似系数二值化图像轮廓')%显示近似系数
subplot(222), imshow(A2, []);title('水平细节分量二值化图像轮廓')%显示水平细节分量
subplot(223), imshow(A3, []);title('垂直细节分量二值化图像轮廓')%显示垂直细节分量
subplot(224), imshow(A4, []);title('对角细节分量二值化图像轮廓')%显示对角细节分量
