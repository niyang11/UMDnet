clear all
close all  

%sw=2500;
l_t2=8192;
% l_t2=1024;
% t2 = 0:1/(sw*2):1.6382; 
sw=8000;
t2 = 0:1/sw:(l_t2/sw);
% t2 = 0:1/sw:0.0125;
% t2 = 0:1/(sw):1.6382;
fn=8192;
num=1000;
% f1=zeros(20,l_t2);
FFT=zeros(num*1,fn);
FFTN=zeros(num*1,fn);
data=zeros(num*1,fn*2);
step=1;
ppm=linspace(-sw/2,sw/2,2*fn);  
ppm1=linspace(-sw/2,sw/2,fn);  
for i=1:step:num
    for n=1:20
        a=0.01+(0.9-0.01)*rand(1);
%         a=0.01+(0.02-0.01)*rand(1);
%         a=0.1;
%         a=0.1;
       w=(-2500+(2500-(-2500))*rand(1))*2*pi;
%         r=0.003;
        r=0.003+(0.3-0.003)*rand(1);
        p=(0+(1-0)*rand(1))<0.5;
%          p=1;
        f1(n,:)=p*create_f2(a, w, r, t2);
    end

    for m=i:min(i+3,num)
%         n=m-i+1;
        fid=sum(f1,1);
        fidfft=fftshift(fft(fid,fn));
        fidfft_real_max=max(abs(fidfft));
%         fidfft_normalize=fidfft/fidfft_real_max;
        fidfft_normalize=normalize_signal(fidfft);
        fid2=(ifft(fidfft_normalize));
        snr=(8e-5)+((2e-4)-(8e-5))*rand(1);
        noise=snr*(randn(1,fn)+1i*randn(1,fn));
        fid2_noise=fid2+noise;

        fidfft2=(fft(fid2_noise,fn));

        fidfft_normalize2=fidfft2;
        FFT(m,:)=real(fidfft_normalize);
        FFTN(m,:)=real(fidfft_normalize2);

% 

%         FFT = (FFT ) / max(FFT) ;  % 归一化 FFT
%         FFTN = (FFTN ) / max(FFTN);  % 归一化 FFTN
       
%          FFT(m,:)=normalize_signal(fid2);
%          FFTN(m,:)=normalize_signal(fid2_noise);

%         data(m,:)=cat(2,FFT,FFTN);
%         NOISE(:,m)=real(noise);
    end   
end
% a=FFT;
% b=real(fft(a));%input
% c=fft(real(a));
% e=real(fft(real(ifft(b))));
% 20000SS
% figure(3),plot(b);
% figure(4),plot(e);
% real_ideal_sp=real(FFT);
% max_idl_value=max(abs(real_ideal_sp));
% FFT=real_ideal_sp./max_idl_value;
% 
% real_ideal_sp=real(FFTN);
% max_idl_value=max(abs(real_ideal_sp));
% FFTN=real_ideal_sp./max_idl_value;

% dataset=zeros(2,fn,num*4);
% dataset(1,:,:)=FFT;
% dataset(2,:,:)=FFTN;

% real_ideal_sp=real(NOISE);
% max_idl_value=max(abs(real_ideal_sp));
% NOISE=real_ideal_sp./max_idl_value;

figure(1),plot(FFT(1,:));
figure(2),plot(FFTN(1,:));

% figure(3),plot(real(normalize_signal(fft(FFT(1,:),fn)))); figure(4),plot(real(normalize_signal(fft(FFTN(1,:),fn))));

% c=real(a);
% d=fft(c);
% e=ifft(d);
% f=fft(e);
% g=real(ifft(f));
% h=ifft(real(f));
% b = data(2,:);
% c = data(3,:);
% d = data(4,:);
% noise1=a(8000+8000:8100+8000);
% noise2=b(8000+8000:8100+8000);
% noise3=c(8000+8000:8100+8000);
% noise4=d(8000+8000:8100+8000);
% snr1=max(a)/std(noise1);
% snr2=max(b)/std(noise2);
% snr3=max(c)/std(noise3);
% snr4=max(d)/std(noise4);
  

% figure(3),plot(ppm,b);
% figure(4),plot(ppm,c);
% figure(5),plot(ppm,d);
% FFT=FFT(1,:);
% FFTN=FFTN(1,:);

FFT=FFT';
FFTN=FFTN';

% save('test999.mat','FFT','FFTN');
% save('/home/project/ny/matlab/data/train_data_fre/20000_SS_time_domain1.mat','FFT','FFTN','-v7.3');
save('/home/project/ny/matlab/data/train_data/IST_1000SS.mat','FFT','FFTN','-v7.3');
% save('/home/project/ny/matlab/wave_unet/test_layer.mat','FFT','FFTN');
 % save('/home/project/ny/matlab/wave_unet/test_model.mat','FFT','FFTN');
% save('/mnt/bfeefeff-9dde-46eb-a2f1-4576487a9a01/fangqy/data/simulation_data_SS_8192_5000_snr8e52e4_sw8000.txt','data','-ascii');
% % savepath='/home/fangqy/matlab/train_data';
% h5create('data_sw20000_r0.003.h5',savepath,[2 fn num*4]);
% h5write('data.h5',savepath,dataset);

function signal_normalized = normalize_signal(signal)
    % 计算信号的最大模值
    max_val = max(abs(signal));
    if max_val ~= 0
        signal_normalized = signal / max_val;
    else
        signal_normalized = signal;
    end
end


