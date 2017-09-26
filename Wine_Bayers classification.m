clear all; close all; clc;

  data = dlmread('wine.data');
  data_mean = mean(data);
  % normalize 
%  data(:,2:14) = data(:,2:14) - mean(2:14);
  

  K = 10;
  [nums, features] = size(data);
  index = randperm(nums);          
  data_shuffled = data(index, :);
  data_shuffled = data_shuffled(1:170, :);
  [nums, features] = size(data_shuffled);
  accuracy = zeros(K,1);
  for k=1:K
 
    test_index = [1+(k-1)*nums/10:k*nums/10];
    train_index = [1:(k-1)*nums/10,k*nums/10+1:nums];
  
    train_data = data_shuffled(train_index,:);
    
    test_data = data_shuffled(test_index,2:14);
    test_label = data_shuffled(test_index,1);
    
    classcolumn = train_data(:,1);
    testnums = length(test_data);

    data_class1 = train_data(classcolumn==1,2:14);
    data_class2 = train_data(classcolumn==2,2:14);
    data_class3 = train_data(classcolumn==3,2:14);
    
    mean_class1 = transpose(mean(data_class1));
    mean_class2 = transpose(mean(data_class2));
    mean_class3 = transpose(mean(data_class3));
  
    cov_class1 = covmle(data_class1, mean_class1);
    cov_class2 = covmle(data_class2, mean_class2);
    cov_class3 = covmle(data_class3, mean_class3);

    yhat = zeros(testnums,1);

    for i=1:testnums
      X = transpose(test_data(i,:));
      g1 = -0.5*transpose(X - mean_class1)*inv(cov_class1)*(X- mean_class1) - 0.5*log(det(cov_class1)) + log(1/3);
      g2 = -0.5*transpose(X - mean_class2)*inv(cov_class2)*(X- mean_class2) - 0.5*log(det(cov_class2)) + log(1/3);
      g3 = -0.5*transpose(X - mean_class3)*inv(cov_class3)*(X- mean_class3) - 0.5*log(det(cov_class3)) + log(1/3);
      if g1>=g2 && g1>=g3
        yhat(i,1)=1;
      elseif g2>=g1 && g2>=g3
        yhat(i,1)=2;
      else
        yhat(i,1) = 3;
      end 
     
      if yhat(i,1) == test_label(i,1)
        accuracy(k,1) += 1;
      end
    end
    accuracy(k,1) = accuracy(k,1)/testnums;
  end
  
  fprintf("Classification accuracy\n");
  fprintf("ans = \n");
  disp(accuracy)
  fprintf('average = %5.4f \n\n', sum(accuracy(:,1))/K);
