clc
clear all
close all
warning off
imgsize=[300,300];
[pathstr,~,~]= fileparts(mfilename('fullpath'));
pics1=dir([pathstr,'/faces/*.jpg']);MBS=length(pics1);
stImageSavePath  = strcat(pathstr,'/faceget/');
%%
%图像-读取-预处理-人脸获取—小波提取-Hu不变矩提取-数据库建立
if ~exist('face.mat','file')
disp('数据库生成中');
if MBS > 0                                           %批量读入图片，进行五官检测，再批量检测
for i = 1:MBS
iSaveNum      = int2str(i);
stImagePath   = pics1(i).name;
mImageCurrent = imread(strcat(pathstr,'/faces/',pics1(i).name));%获取人脸
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
myhsv=rgb2hsv(rgb);%对图像进行HSV处理
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
mFaceResult = imresize(mFaceResult,imgsize, 'bilinear');%双线性插值方法
imwrite(mFaceResult,strcat(stImageSavePath,iSaveNum,'.bmp')); 
[cA1, cH1, cV1, cD1] = dwt2(mFaceResult, 'haar');%小波分量提取分量
HU(:,i,1)=Humoment(cA1);
% HU(:,i,2)=Humoment(cH1);
% HU(:,i,3)=Humoment(cV1);
% HU(:,i,4)=Humoment(cD1);
end
end
save face.mat; 
disp('数据库建立完毕');
else
disp('已存在一个face.mat的数据库，如需更新将原有数据库删除重新运行本程序即可')
end
%%
%数据库导入Hu不变矩相似度计算-人脸识别
load('face.mat')
disp('已导入数据库');
[~,l,~]=size(HU);
[fn,pn,fi]=uigetfile('*.*','选择图片');%选择图片
face=imread([pn fn]);%读取原始图像
face = imresize(face,imgsize, 'bilinear');%双线性插值方法
[~,~,sizes]=size(face);
if sizes>2
face=rgb2gray(face);%彩色图像装化为二值图像
else
face =face;
end
figure(1);
imshow(face);%彩色原始图像
title('人脸原始图像')
figure(2);
[cA1, cH1, cV1, cD1] = dwt2(face, 'haar');%小波分量提取分量
subplot(221), imshow(cA1, []);title('近似系数')%显示近似系数
subplot(222), imshow(cH1, []);title('水平细节分量')%显示水平细节分量
subplot(223), imshow(cV1, []);title('垂直细节分量')%显示垂直细节分量
subplot(224), imshow(cD1, []);title('对角细节分量')%显示对角细节分量
figure(3)
A1=im2bw(cA1,0.5);
A2=im2bw(cH1,0.5);
A3=im2bw(cV1,0.5);
A4=im2bw(cD1,0.5);
subplot(221), imshow(A1, []);title('近似系数二值化图像')%显示近似系数
subplot(222), imshow(A2, []);title('水平细节分量二值化图像')%显示水平细节分量
subplot(223), imshow(A3, []);title('垂直细节分量二值化图像')%显示垂直细节分量
subplot(224), imshow(A4, []);title('对角细节分量二值化图像')%显示对角细节分量
figure(4);
img2=bwperim(A2);
img3=bwperim(A3);
img4=bwperim(A4);
subplot(221), imshow(A1, []);title('近似系数二值化图像轮廓')%显示近似系数
subplot(222), imshow(A2, []);title('水平细节分量二值化图像轮廓')%显示水平细节分量
subplot(223), imshow(A3, []);title('垂直细节分量二值化图像轮廓')%显示垂直细节分量
subplot(224), imshow(A4, []);title('对角细节分量二值化图像轮廓')%显示对角细节分量
HUtest(:,:,1)=Humoment(cA1).';
HUtest(:,:,2)=Humoment(cH1).';
HUtest(:,:,3)=Humoment(cV1).';
HUtest(:,:,4)=Humoment(cD1).';
for i=1:l
xsd1(i)=norm(abs(log(HUtest(1:7,1,1)))-abs(log(HU(1:7,i,1)))); %与训练样本进行匹配
% xsd2(i)=norm(abs(log(HUtest(1:7,1,2)))-abs(log(HU(1:7,i,2)))); %与训练样本进行匹配
% xsd3(i)=norm(abs(log(HUtest(1:7,1,3)))-abs(log(HU(1:7,i,3)))); %与训练样本进行匹配
% xsd4(i)=norm(abs(log(HUtest(1:7,1,4)))-abs(log(HU(1:7,i,4)))); %与训练样本进行匹配
end
xsd=xsd1;
answer=find(xsd==min(min(xsd)));
[pathstr,name,ext]= fileparts(mfilename('fullpath'));
fname=strcat(pathstr,'\faces\',pics1(answer).name);
fprintf('识别结果为%s\n',fname);
rgb=imread(fname);%
figure(5);
imshow(rgb);
title('识别结果');%显示图像 