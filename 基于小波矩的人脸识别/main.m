clc
clear all
close all
warning off
imgsize=[300,300];
[pathstr,~,~]= fileparts(mfilename('fullpath'));
pics1=dir([pathstr,'/faces/*.jpg']);MBS=length(pics1);
stImageSavePath  = strcat(pathstr,'/faceget/');
%%
%ͼ��-��ȡ-Ԥ����-������ȡ��С����ȡ-Hu�������ȡ-���ݿ⽨��
if ~exist('face.mat','file')
disp('���ݿ�������');
if MBS > 0                                           %��������ͼƬ��������ټ�⣬���������
for i = 1:MBS
iSaveNum      = int2str(i);
stImagePath   = pics1(i).name;
mImageCurrent = imread(strcat(pathstr,'/faces/',pics1(i).name));%��ȡ����
rgb = mImageCurrent;
R=rgb(:,:,1);
G=rgb(:,:,2);
B=rgb(:,:,3);
maxR=im2double(max(max(R)));
maxG=im2double(max(max(G)));
maxB=im2double(max(max(B)));
k=(maxR+maxG+maxB)/3;
if k<0.9
rgb = imadjust(rgb,[0 0 0; k k k],[0 0 0;1 1 1],0.7);
end
myhsv=rgb2hsv(rgb);%��ͼ�����HSV����
H=myhsv(:,:,1);
S=myhsv(:,:,2);
V=myhsv(:,:,3);
maxH=im2double(max(max(H)));
maxS=im2double(max(max(S)));
maxV=im2double(max(max(V)));
k=(maxH+maxS+maxV)/3;
if k<0.9
rgb = imadjust(rgb,[0 0 0; k k k],[0 0 0;1 1 1],0.7);
end
mFaceResult   = face_segment(rgb);
mFaceResult = imresize(mFaceResult,imgsize, 'bilinear');%˫���Բ�ֵ����
imwrite(mFaceResult,strcat(stImageSavePath,iSaveNum,'.bmp')); 
[cA1, cH1, cV1, cD1] = dwt2(mFaceResult, 'haar');%С��������ȡ����
HU(:,i,1)=Humoment(cA1);
% HU(:,i,2)=Humoment(cH1);
% HU(:,i,3)=Humoment(cV1);
% HU(:,i,4)=Humoment(cD1);
end
end
save face.mat; 
disp('���ݿ⽨�����');
else
disp('�Ѵ���һ��face.mat�����ݿ⣬������½�ԭ�����ݿ�ɾ���������б����򼴿�')
end
%%
%���ݿ⵼��Hu��������ƶȼ���-����ʶ��
load('face.mat')
disp('�ѵ������ݿ�');
[~,l,~]=size(HU);
[fn,pn,fi]=uigetfile('*.*','ѡ��ͼƬ');%ѡ��ͼƬ
face=imread([pn fn]);%��ȡԭʼͼ��
face = imresize(face,imgsize, 'bilinear');%˫���Բ�ֵ����
[~,~,sizes]=size(face);
if sizes>2
face=rgb2gray(face);%��ɫͼ��װ��Ϊ��ֵͼ��
else
face =face;
end
figure(1);
imshow(face);%��ɫԭʼͼ��
title('����ԭʼͼ��')
figure(2);
[cA1, cH1, cV1, cD1] = dwt2(face, 'haar');%С��������ȡ����
subplot(221), imshow(cA1, []);title('����ϵ��')%��ʾ����ϵ��
subplot(222), imshow(cH1, []);title('ˮƽϸ�ڷ���')%��ʾˮƽϸ�ڷ���
subplot(223), imshow(cV1, []);title('��ֱϸ�ڷ���')%��ʾ��ֱϸ�ڷ���
subplot(224), imshow(cD1, []);title('�Խ�ϸ�ڷ���')%��ʾ�Խ�ϸ�ڷ���
figure(3)
A1=im2bw(cA1,0.5);
A2=im2bw(cH1,0.5);
A3=im2bw(cV1,0.5);
A4=im2bw(cD1,0.5);
subplot(221), imshow(A1, []);title('����ϵ����ֵ��ͼ��')%��ʾ����ϵ��
subplot(222), imshow(A2, []);title('ˮƽϸ�ڷ�����ֵ��ͼ��')%��ʾˮƽϸ�ڷ���
subplot(223), imshow(A3, []);title('��ֱϸ�ڷ�����ֵ��ͼ��')%��ʾ��ֱϸ�ڷ���
subplot(224), imshow(A4, []);title('�Խ�ϸ�ڷ�����ֵ��ͼ��')%��ʾ�Խ�ϸ�ڷ���
figure(4);
img2=bwperim(A2);
img3=bwperim(A3);
img4=bwperim(A4);
subplot(221), imshow(A1, []);title('����ϵ����ֵ��ͼ������')%��ʾ����ϵ��
subplot(222), imshow(A2, []);title('ˮƽϸ�ڷ�����ֵ��ͼ������')%��ʾˮƽϸ�ڷ���
subplot(223), imshow(A3, []);title('��ֱϸ�ڷ�����ֵ��ͼ������')%��ʾ��ֱϸ�ڷ���
subplot(224), imshow(A4, []);title('�Խ�ϸ�ڷ�����ֵ��ͼ������')%��ʾ�Խ�ϸ�ڷ���
HUtest(:,:,1)=Humoment(cA1).';
HUtest(:,:,2)=Humoment(cH1).';
HUtest(:,:,3)=Humoment(cV1).';
HUtest(:,:,4)=Humoment(cD1).';
for i=1:l
xsd1(i)=norm(abs(log(HUtest(1:7,1,1)))-abs(log(HU(1:7,i,1)))); %��ѵ����������ƥ��
% xsd2(i)=norm(abs(log(HUtest(1:7,1,2)))-abs(log(HU(1:7,i,2)))); %��ѵ����������ƥ��
% xsd3(i)=norm(abs(log(HUtest(1:7,1,3)))-abs(log(HU(1:7,i,3)))); %��ѵ����������ƥ��
% xsd4(i)=norm(abs(log(HUtest(1:7,1,4)))-abs(log(HU(1:7,i,4)))); %��ѵ����������ƥ��
end
xsd=xsd1;
answer=find(xsd==min(min(xsd)));
[pathstr,name,ext]= fileparts(mfilename('fullpath'));
fname=strcat(pathstr,'\faces\',pics1(answer).name);
fprintf('ʶ����Ϊ%s\n',fname);
rgb=imread(fname);%
figure(5);
imshow(rgb);
title('ʶ����');%��ʾͼ�� 