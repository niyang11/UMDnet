
function [f2, s2] = create_f2(a2, w2, r2, t2) 
for i = 1:length(a2)
    f2(:, i) = a2(i)*exp(1i*w2(i)*t2).*exp(-t2/r2);
end

% function [f2, s2] = create_f2(a2, w2, r2, t2) 
%     f2 = zeros(length(t2), length(a2));
%       s2 = zeros(1, length(a2));
%     
%     for i = 1:length(a2)
%         global initial_height;
%         s2(i) = initial_height + (i-1) * height_increment; % 逐渐递增的幅度
%         f2(:, i) = s2(i) * exp(1i * w2(i) * t2) .* exp(-t2 / r2); % 使用单一的衰减因子 r2
%         
%     end
% end