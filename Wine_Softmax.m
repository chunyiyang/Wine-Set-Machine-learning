clear all; close all; clc;
data = dlmread('wine.data');
normalize = max(data) - min(data);
data(:,2:14) = data(:,2:14)./normalize(2:14);
data = data(1:170,:);
function g = sigmoid(z)
g = exp(z)/(1 + exp(z));
end

function sfx = softmax(x,w)
  sumexp = sum(exp(x*w));
  sfx = exp(x*w)/sumexp;
end

function h = H(true_label, sfx)
  h = - sum(true_label.*log(sfx));
end
[nums, features] = size(data);

K = 10 ;
accarray = zeros(K,1);
loop_count = 1500;
cost = zeros(loop_count,1);
index = randperm(nums);          
data_shuffled = data(index, :);
batchsize = int32(nums/10);
for k=1:10
  test_index = [1+(k-1)*batchsize:k*batchsize];
  train_index = [1:(k-1)*batchsize,k*batchsize+1:nums];
  
  train_data = data_shuffled(train_index,2:14);
  train_label = data_shuffled(train_index, 1);
  
  test_data = data_shuffled(test_index,2:14);
  test_label = data_shuffled(test_index,1);
  
  l = length(test_data);
  
  test_label_matrix = zeros(l,3);
  for p = 1:length(train_data)
    train_label_matrix(p, train_label(p,1)) = 1;
  end

%  normalize = max(train_data) - min(train_data);
%  train_data = train_data./normalize;
  
  train_data = [repmat(1, length(train_data),1) train_data];
  test_data = [repmat(1, length(test_data),1) test_data];

  [m n] = size(train_data);
  % softmax
  w = rand(n, 3);
  
  yhat = zeros(m,1);
  nu = 0.01;
  for loop=1:loop_count
    % softmax
    step_sum = zeros(n,3);
    cost_itr = 0;
    
    for i=1:m
      sfx = softmax(train_data(i,:),w);
      for j=1:n
       step_sum(j,:) += (sfx - train_label_matrix(i,:))*train_data(i,j);
      end 
      cost_itr = H(train_label_matrix(i,:), sfx);
    end
    temp_w = w - nu*step_sum/m;
    w = temp_w;
    
    cost(loop,1) = -cost_itr;
  end
  % start training part
 
  yhat = zeros(l,1);
  for i=1:l
      result = softmax(test_data(i,:), w);
      max = 0;
      for j = 1:3
        if (result(j)>max)
          max = result(j);
          yhat(i,1) = j;
        endif
      end 
      
      if(yhat(i,1) == test_label(i,1))
        accarray(k,1)+=1;
      endif  
   end
  
end
accarray = accarray/l;
fprintf("Classification accuracy\n");
fprintf("ans = \n");
disp(accarray);
average = (sum(accarray))/K;
fprintf("Average accuracy = %5.4f \n", average);
plot(cost);
xlabel('Training iterations');
ylabel('Cost function J');
 
