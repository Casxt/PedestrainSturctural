addpath('/mnt/hdfs_fuse/user/zhangkai/pa-100k');
load('annotation.mat');
attributes(length(attributes)+1)={'Male'};

test_label(:,length(attributes)) = not(test_label(:,1));
train_label(:,length(attributes)) = not(train_label(:,1));
val_label(:,length(attributes)) = not(val_label(:,1));
save('newAnnotation.mat', 'attributes', 'test_images_name', 'test_label', 'train_images_name', 'train_label', 'val_images_name', 'val_label');


label = fopen('../label.names','w');
for i = 1:length(attributes)
    fprintf(label,'%s\n', attributes{i});
end
fclose(label);

data = fopen('../data-100k-27class.txt', 'w');
writeDataLabel( attributes, train_images_name, train_label, data );
writeDataLabel( attributes, val_images_name, val_label, data );
writeDataLabel( attributes, test_images_name, val_label, data );
fclose(data);