function [attributes, train_label, test_label, val_label] = deleteAttribute(name, attributes, train_label, test_label, val_label)
    index = 0;
    for i = 1:length(attributes)
        if strcmpi(name, attributes{i}) == 1
            index = i;
            break;
        end
    end
    if not(index == 0)
        attributes(index) = [];
        train_label(:,i) = [];
        test_label(:,i) = [];
        val_label(:,i) = [];
        sprintf('%s has been delete\n',name);
    else 
        sprintf('%s not found\n',name);
    end
end