function [ ] = writeDataLabel( attributes, train_images_name, train_label, data )
%WRITEDATALABEL Summary of this function goes here
%   Detailed explanation goes here

    label_num = length(attributes);
    for i = 1:length(train_images_name)
        fprintf(data,'/mnt/hdfs_fuse/user/zhangkai/pa-100k/release_data/release_data/%s', train_images_name{i});
        for label_index = 1:label_num
            if train_label(i, label_index) ~= 0
                fprintf(data,',%s',attributes{label_index});
            end
        end
        fprintf(data,'\n');
        disp(train_images_name{i});
    end

end

