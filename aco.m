%%%%%%%%%%%%%%%%%%%��ʼ��%%%%%%%%%%%%%%%%%

t0 = clock;   
city=[0,1;1,0;1,2;2,0;2,2;3,0;3,2;4,0;4,2;5,0;5,2;6,0;6,2;7,0;7,2;8,0;8,2;9,0;9,2;10,0;10,2;11,0;11,2];

labels=xlsread('D:\matlab.m\shuju\��ǩ.xlsx'); 
jiazha=importdata('D:\matlab.m\shuju\jiazha.txt');
qikong=importdata('D:\matlab.m\shuju\qikong.txt');
liewen=importdata('D:\matlab.m\shuju\liewen.txt');
weihantou=importdata('D:\matlab.m\shuju\weihantou.txt');
weironghe=importdata('D:\matlab.m\shuju\weironghe.txt');
datas_normal=[jiazha;qikong;liewen;weihantou;weironghe];

datas_normal=mapminmax(datas_normal',0,1);
datas_normal=datas_normal';

n = size(city,1);                                                                      %���о����ʼ��

xunlian = zeros(105,11);                                                    

m=30;                                                                                        %��������

alpha = 0.5;                                                                                 % ��Ϣ����Ҫ�̶�����

% beta = 5;                                                                                  % ����������Ҫ�̶�����

v = 0.1;                                                                                     % ��Ϣ�ػӷ�����

Q = 0.5;                                                                                     % ��Ϣ���ӳ�ϵ��


T= ones(23,23);                                                                         % ��Ϣ�ؾ���,��ʼֵΪ1

Table = zeros(m,12);                                                               % ·����¼��

iter = 1;                                                                                    % ����������ֵ

iter_max = 50;                                                                     % ���������� 

best_route = zeros(iter_max,12);                                       % �������·��       

best_length = zeros(iter_max,1);                                     % �������·���ĳ���  

h=zeros(m,1);%���ڴ��ÿ�����ϵ��Ӽ�����

%%

while iter<=iter_max

    

                        % �������ÿֻ���ϵ�������

                          start = ones(m,1);


                          Table(:,1) = start; %����ÿֻ���ϵ���ʼ·����ʹtable·������ĵ�һ��Ϊstart���ֵ,table�ڴ�֮ǰ��û��ָ����ʼֵ��tableΪm*n�ľ���

                          city_index=1:n;%���ó��б��

                      for i = 1:m

                          % �������·��ѡ��

                         for j = 2:12

                           
                            
                             allow =[city_index(2*j-2),city_index(2*j-1)];   % ɸѡ��δ���ʵĳ��м��ϣ�allow�ǳ�ȥtabu�����������

                             P = zeros(1,length(allow));

                             % �����������е�ת�Ƹ���

                             for k = 1:length(allow)

                                 P(k) = T(Table(i,j-1),allow(k))^alpha; %T��ǰΪones��n��n������

                             end

                             P = P/sum(P);

                             % ���̶ķ�ѡ����һ�����ʳ���

                            Pc = cumsum(P);     %�μ�˵��2(����ײ�)

                            target_index = find(Pc >= rand); 

                            target = allow(target_index(1));

                            Table(i,j) = target;

                         end

                      end

 

                          % ����������ϵ�·������

                                  Length = zeros(m,1);

                                  for i = 1:m

                                      Route = [Table(i,:)];%route��һ��1*��n+1�����������������ÿֻ���ϵĻ�··��
                                      Route=rem(Route,2);
                                      Route=Route(1,2:12);
                                      for j=1:11
                                          if Route(j)==0
                                          xunlian(:,j)=datas_normal(:,j);
                                          end
                                      end
                                      xunlian(:,find(sum(abs(xunlian),1)==0))=[];%�γɽ�ά���Ӽ�
                                      h(i,1)=size(xunlian,2);%��ȡ�Ӽ���������
                                      
                                      kk =5;%Ԥ�����ݷֳ�ʮ��
                                      sum_accuracy_svm = 0;
                                    [mm,nn] = size(xunlian);
                                    %������֤,ʹ��ʮ�۽�����֤  Kfold  
                                    %indicesΪ n ��һ�����ݣ���ʾÿ��ѵ����������k�����ݵ���һ��
                                    indices = crossvalind('Kfold',mm,kk);

                                    for ii = 1:kk
                                        test_indic = (indices == ii);
                                        train_indic = ~test_indic;
                                        train_datas = xunlian(train_indic,:);%�ҳ�ѵ���������ǩ
                                        train_labels = labels(train_indic,:);
                                        test_datas = xunlian(test_indic,:);%�ҳ������������ǩ
                                        test_labels = labels(test_indic,:);
                              
                                        classifer = svmtrain(train_labels,train_datas,'-c 1 -g 0.7');%ѵ��ģ��
%                                         classifer = svmtrain(train_labels,train_datas,'-c 0.83 -g 3.09');%ѵ��ģ��
                                        [predict_label, accuracy, dec_values]  = svmpredict(test_labels, test_datas,classifer);%����
                                    %     accuracy_svm = accuracy_svm+accuracy(1,1);%׼ȷ��
                                        accuracy_svm = length(find(predict_label == test_labels))/length(test_labels)%׼ȷ��
                                        sum_accuracy_svm = sum_accuracy_svm + accuracy_svm;

                                    end

                                    %��ƽ��׼ȷ��
                                    mean_accuracy_svm = sum_accuracy_svm / kk;
                                    Length(i) = mean_accuracy_svm;%length��һ��m*1���������������ÿֻ���ϵ�·�߳���
                                    
                                    xunlian = zeros(105,11);%����ѵ���Ӽ�

                                  end   

             %������·�ߺ;������            

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

                            % ������Ϣ��

                          Delta_T= zeros(23,23);

                          % ������ϼ���

                          for i = 1:m

                              % ������м���

                              Route = Table(i,2:12);

                              for j = 1:10

                                  Delta_T(Route(j),Route(j+1)) = Delta_T(Route(j),Route(j+1)) + Q*Length(i)*(1-h(i,1)/11);

                              end

                          end

                          T= (1-v) * T + Delta_T;

                                 % ����������1�������·����¼��

                        iter = iter + 1;

                        Table = zeros(m,12);              

end

%--------------------------------------------------------------------------

%% �����ʾ

shortest_route=best_route(end,:);                 %ѡ����̵�·���еĵ�

short_length=best_length(end);

Time_Cost=etime(clock,t0);

disp(['��̾���:' num2str(short_length)]);

disp(['���·��:' num2str([shortest_route shortest_route(1)])]);

disp(['����ִ��ʱ��:' num2str(Time_Cost) '��']);



%--------------------------------------------------------------------------

%% ��ͼ

figure(1)

%��������ͼ������

plot([city(shortest_route,1)], [city(shortest_route,2)],'o-');

% for i = 1:size(city,1)
% 
%     %��ÿ�����н��б��
% 
%     text(city(i,1),city(i,2),['   ' num2str(i)]);
% 
% end

xlabel('����ֵ','FontSize',14)

ylabel('����ֵѡȡ״̬','FontSize',14)
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


% title(['��Ⱥ�㷨���Ż�·��(��̾��룩:' num2str(short_length) ''])

 

figure(2)

%������������

plot(1:iter_max,best_length,'b')

xlabel('��������')

ylabel('ʶ����(%)')

title('������������')