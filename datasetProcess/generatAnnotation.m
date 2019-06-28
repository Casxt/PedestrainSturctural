addpath('/mnt/hdfs_fuse/user/zhangkai/pa-100k');
load('annotation.mat');
deleteedAttibute = {};%'Female','AgeOver60','Age18-60','AgeLess18','Glasses','boots'
for i = 1:length(deleteedAttibute)
   [attributes, train_label, test_label, val_label] = deleteAttribute(deleteedAttibute{i}, attributes, train_label, test_label, val_label);
end
save('newAnnotation.mat', 'attributes', 'test_images_name', 'test_label', 'train_images_name', 'train_label', 'val_images_name', 'val_label');


label = fopen('../label.names','w');
for i = 1:length(attributes)
    fprintf(label,'%s\n', attributes{i});
end
fclose(label);

data = fopen('../data.txt', 'w');
writeDataLabel( attributes, train_images_name, train_label, data );
writeDataLabel( attributes, val_images_name, val_label, data );
fclose(data);