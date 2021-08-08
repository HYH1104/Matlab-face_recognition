clc
clear all
close all
warning off
imgsize=[300,300];
[fn,pn,fi]=uigetfile('*.*','ѡ��ͼƬ');%ѡ��ͼƬ
rgb=imread([pn fn]);%��ȡԭʼͼ��
figure(1);
imshow(rgb);%��ɫԭʼͼ��
title('����ԭʼͼ��')
disp('ԭʼͼ���С:')
disp(size(rgb))
figure(2);
R=rgb(:,:,1);
G=rgb(:,:,2);
B=rgb(:,:,3);
subplot(131), imshow(R);title('R��ǿ��ͼ��')
subplot(132), imshow(G);title('G��ǿ��ͼ��')
subplot(133), imshow(B);title('B��ǿ��ͼ��')
maxR=im2double(max(max(R)));
maxG=im2double(max(max(G)));
maxB=im2double(max(max(B)));
k=(maxR+maxG+maxB)/3;
if k<0.9
rgb = imadjust(rgb,[0 0 0; k k k],[0 0 0;1 1 1],0.7);
end
figure(3)
imshow(rgb);
title('rgbǿ�ȴ���ͼ��')
figure(4)
myhsv=rgb2hsv(rgb);%��ͼ�����HSV����
H=myhsv(:,:,1);
S=myhsv(:,:,2);
V=myhsv(:,:,3);
subplot(131), imshow(H);title('H��ǿ��ͼ��')
subplot(132), imshow(S);title('S��ǿ��ͼ��')
subplot(133), imshow(V);title('V��ǿ��ͼ��')
maxH=im2double(max(max(H)));
maxS=im2double(max(max(S)));
maxV=im2double(max(max(V)));
k=(maxH+maxS+maxV)/3;
if k<0.9
rgb = imadjust(rgb,[0 0 0; k k k],[0 0 0;1 1 1],0.7);
end
figure(5)
imshow(rgb);
title('hsvǿ�ȴ���ͼ��')
mFaceResult   = face_segment(rgb);
figure(6);
imshow(mFaceResult);%��ɫԭʼͼ��
title('��������ͼ��')
disp('��һ������ͼ���С:')
disp(size(mFaceResult))
figure(7);
[cA1, cH1, cV1, cD1] = dwt2(mFaceResult, 'haar');%С��������ȡ����
HUtest(:,1)=Humoment(cA1).';
HUtest(:,2)=Humoment(cH1).';
HUtest(:,3)=Humoment(cV1).';
HUtest(:,4)=Humoment(cD1).';
subplot(221), imshow(cA1, []);title('����ϵ��')%��ʾ����ϵ��
subplot(222), imshow(cH1, []);title('ˮƽϸ�ڷ���')%��ʾˮƽϸ�ڷ���
subplot(223), imshow(cV1, []);title('��ֱϸ�ڷ���')%��ʾ��ֱϸ�ڷ���
subplot(224), imshow(cD1, []);title('�Խ�ϸ�ڷ���')%��ʾ�Խ�ϸ�ڷ���
figure(8)
A1=im2bw(cA1,0.5);
A2=im2bw(cH1,0.5);
A3=im2bw(cV1,0.5);
A4=im2bw(cD1,0.5);
subplot(221), imshow(A1, []);title('����ϵ����ֵ��ͼ��')%��ʾ����ϵ��
subplot(222), imshow(A2, []);title('ˮƽϸ�ڷ�����ֵ��ͼ��')%��ʾˮƽϸ�ڷ���
subplot(223), imshow(A3, []);title('��ֱϸ�ڷ�����ֵ��ͼ��')%��ʾ��ֱϸ�ڷ���
subplot(224), imshow(A4, []);title('�Խ�ϸ�ڷ�����ֵ��ͼ��')%��ʾ�Խ�ϸ�ڷ���
figure(9);
img2=bwperim(A2);
img3=bwperim(A3);
img4=bwperim(A4);
subplot(221), imshow(A1, []);title('����ϵ����ֵ��ͼ������')%��ʾ����ϵ��
subplot(222), imshow(A2, []);title('ˮƽϸ�ڷ�����ֵ��ͼ������')%��ʾˮƽϸ�ڷ���
subplot(223), imshow(A3, []);title('��ֱϸ�ڷ�����ֵ��ͼ������')%��ʾ��ֱϸ�ڷ���
subplot(224), imshow(A4, []);title('�Խ�ϸ�ڷ�����ֵ��ͼ������')%��ʾ�Խ�ϸ�ڷ���
