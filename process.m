clear all;
clc;
load('Y:\dataset\Decoding_Data\hc_data_raw.mat');
load('Y:\dataset\Decoding_Data\m1_data_raw.mat');
nn=size(spike_times,1);
len=length(vel_times);
binlen=0.05;
begin_time=vel_times(1,1);
end_time=vel_times(end,1);
% rate_num=ceil(length(spike_times)/binlen*1000);
spike_rate=begin_time:binlen:end_time;
spike_rate=repmat(spike_rate,length(spike_times),1);
spike_rate=zeros(size(spike_rate,1),size(spike_rate,2));
for i=1:nn
    len_spike_time=length(spike_times{i});
    for j=1:len_spike_time
        stamp=ceil(spike_times{i}(j)/binlen-begin_time*binlen);
        if(stamp>0 && stamp<length(spike_rate))
            spike_rate(i,stamp)=spike_rate(i,stamp)+1;
            disp(stamp);
        end
    end
end


    