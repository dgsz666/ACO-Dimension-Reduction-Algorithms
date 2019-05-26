%%%%%%%%%%%%%%%%%%%初始化%%%%%%%%%%%%%%%%%

t0 = clock;   
city=[0,1;1,0;1,2;2,0;2,2;3,0;3,2;4,0;4,2;5,0;5,2;6,0;6,2;7,0;7,2;8,0;8,2;9,0;9,2;10,0;10,2;11,0;11,2];

labels=xlsread('D:\matlab.m\shuju\标签.xlsx'); 
jiazha=importdata('D:\matlab.m\shuju\jiazha.txt');
qikong=importdata('D:\matlab.m\shuju\qikong.txt');
liewen=importdata('D:\matlab.m\shuju\liewen.txt');
weihantou=importdata('D:\matlab.m\shuju\weihantou.txt');
weironghe=importdata('D:\matlab.m\shuju\weironghe.txt');
datas_normal=[jiazha;qikong;liewen;weihantou;weironghe];

datas_normal=mapminmax(datas_normal',0,1);
datas_normal=datas_normal';

n = size(city,1);                                                                      %城市距离初始化

xunlian = zeros(105,11);                                                    

m=30;                                                                                        %蚂蚁数量

alpha = 0.5;                                                                                 % 信息素重要程度因子

% beta = 5;                                                                                  % 启发函数重要程度因子

v = 0.1;                                                                                     % 信息素挥发因子

Q = 0.5;                                                                                     % 信息因子常系数


T= ones(23,23);                                                                         % 信息素矩阵,初始值为1

Table = zeros(m,12);                                                               % 路径记录表

iter = 1;                                                                                    % 迭代次数初值

iter_max = 50;                                                                     % 最大迭代次数 

best_route = zeros(iter_max,12);                                       % 各代最佳路径       

best_length = zeros(iter_max,1);                                     % 各代最佳路径的长度  

h=zeros(m,1);%用于存放每次蚂蚁的子集个数

%%

while iter<=iter_max

    

                        % 随机产生每只蚂蚁的起点城市

                          start = ones(m,1);


                          Table(:,1) = start; %设置每只蚂蚁的起始路径，使table路径矩阵的第一列为start里的值,table在此之前并没有指定初始值，table为m*n的矩阵

                          city_index=1:n;%设置城市编号

                      for i = 1:m

                          % 逐个城市路径选择

                         for j = 2:12

                           
                            
                             allow =[city_index(2*j-2),city_index(2*j-1)];   % 筛选出未访问的城市集合，allow是除去tabu以外的行向量

                             P = zeros(1,length(allow));

                             % 计算相连城市的转移概率

                             for k = 1:length(allow)

                                 P(k) = T(Table(i,j-1),allow(k))^alpha; %T此前为ones（n，n）矩阵

                             end

                             P = P/sum(P);

                             % 轮盘赌法选择下一个访问城市

                            Pc = cumsum(P);     %参加说明2(程序底部)

                            target_index = find(Pc >= rand); 

                            target = allow(target_index(1));

                            Table(i,j) = target;

                         end

                      end

 

                          % 计算各个蚂蚁的路径距离

                                  Length = zeros(m,1);

                                  for i = 1:m

                                      Route = [Table(i,:)];%route是一个1*（n+1）的行向量，存放着每只蚂蚁的回路路径
                                      Route=rem(Route,2);
                                      Route=Route(1,2:12);
                                      for j=1:11
                                          if Route(j)==0
                                          xunlian(:,j)=datas_normal(:,j);
                                          end
                                      end
                                      xunlian(:,find(sum(abs(xunlian),1)==0))=[];%形成降维后子集
                                      h(i,1)=size(xunlian,2);%获取子集特征个数
                                      
                                      kk =5;%预将数据分成十份
                                      sum_accuracy_svm = 0;
                                    [mm,nn] = size(xunlian);
                                    %交叉验证,使用十折交叉验证  Kfold  
                                    %indices为 n 行一列数据，表示每个训练样本属于k份数据的哪一份
                                    indices = crossvalind('Kfold',mm,kk);

                                    for ii = 1:kk
                                        test_indic = (indices == ii);
                                        train_indic = ~test_indic;
                                        train_datas = xunlian(train_indic,:);%找出训练数据与标签
                                        train_labels = labels(train_indic,:);
                                        test_datas = xunlian(test_indic,:);%找出测试数据与标签
                                        test_labels = labels(test_indic,:);
                              
                                        classifer = svmtrain(train_labels,train_datas,'-c 1 -g 0.7');%训练模型
%                                         classifer = svmtrain(train_labels,train_datas,'-c 0.83 -g 3.09');%训练模型
                                        [predict_label, accuracy, dec_values]  = svmpredict(test_labels, test_datas,classifer);%测试
                                    %     accuracy_svm = accuracy_svm+accuracy(1,1);%准确率
                                        accuracy_svm = length(find(predict_label == test_labels))/length(test_labels)%准确率
                                        sum_accuracy_svm = sum_accuracy_svm + accuracy_svm;

                                    end

                                    %求平均准确率
                                    mean_accuracy_svm = sum_accuracy_svm / kk;
                                    Length(i) = mean_accuracy_svm;%length是一个m*1的列向量，存放着每只蚂蚁的路线长度
                                    
                                    xunlian = zeros(105,11);%重置训练子集

                                  end   

             %对最优路线和距离更新            

                   if iter == 1

                      [min_length,min_index] = max(Length);

                      best_length(iter) = min_length;  

                      best_route(iter,:) = Table(min_index,:);

                  else

                      [min_length,min_index] = max(Length);

                           if min_length>best_length(iter-1)

                                     best_length(iter)=min_length;

                                     best_route(iter,:)=Table(min_index,:);

                           else

                                    best_length(iter)=best_length(iter-1);

                                    best_route(iter,:)=best_route(iter-1,:);

                           end 

                   end

                            % 更新信息素

                          Delta_T= zeros(23,23);

                          % 逐个蚂蚁计算

                          for i = 1:m

                              % 逐个城市计算

                              Route = Table(i,2:12);

                              for j = 1:10

                                  Delta_T(Route(j),Route(j+1)) = Delta_T(Route(j),Route(j+1)) + Q*Length(i)*(1-h(i,1)/11);

                              end

                          end

                          T= (1-v) * T + Delta_T;

                                 % 迭代次数加1，并清空路径记录表

                        iter = iter + 1;

                        Table = zeros(m,12);              

end

%--------------------------------------------------------------------------

%% 结果显示

shortest_route=best_route(end,:);                 %选出最短的路径中的点

short_length=best_length(end);

Time_Cost=etime(clock,t0);

disp(['最短距离:' num2str(short_length)]);

disp(['最短路径:' num2str([shortest_route shortest_route(1)])]);

disp(['程序执行时间:' num2str(Time_Cost) '秒']);



%--------------------------------------------------------------------------

%% 绘图

figure(1)

%采用连线图画起来

plot([city(shortest_route,1)], [city(shortest_route,2)],'o-');

% for i = 1:size(city,1)
% 
%     %对每个城市进行标号
% 
%     text(city(i,1),city(i,2),['   ' num2str(i)]);
% 
% end

xlabel('特征值','FontSize',14)

ylabel('特征值选取状态','FontSize',14)
set(gca, 'YTick', [1],'FontSize',11); 

set(gca,'xaxislocation','bottom' ,'XTick', [1:1:11]); 
set(gca,'xaxislocation','bottom','XTickLabel',{'2','4','6','8','10','12','14','16','18','20','22'},'FontSize',11);
box off;
ylim([0,2]); 
axes; 
plot([city(shortest_route,1)], [city(shortest_route,2)],'o-');
set(gca,'xaxislocation','top' ,'XTick', [1:1:11]); 
set(gca,'xaxislocation','top','XTickLabel',{'3','5','7','9','11','13','15','17','19','21','23'},'FontSize',11);
set(gca, 'YTick', [1],'FontSize',11); 
box off;
ylim([0,2]); 


% title(['蚁群算法最优化路径(最短距离）:' num2str(short_length) ''])

 

figure(2)

%画出收敛曲线

plot(1:iter_max,best_length,'b')

xlabel('迭代次数')

ylabel('识别率(%)')

title('迭代收敛曲线')