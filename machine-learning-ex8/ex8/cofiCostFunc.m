function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
% [i_all,j_all]=find(R==1);
% items=sum(sum(R));
% for i=1:items
%    J=J+(Theta(j_all(i),:)*X(i_all(i),:)'-Y(i_all(i),j_all(i))).^2;
% end
J=0.5*sum(sum((R.*((Theta*X')'-Y)).^2));
reg=(lambda/2)*(sum(sum(Theta.^2))+sum(sum(X.^2)));
J=J+reg;

for i=1:num_movies
    %Find users that have rated the movie
    idx=find(R(i,:)==1);
    %Get thetas for those users
    Theta_t=Theta(idx,:);
    %Get ratings for the movie, for those users
    Y_t=Y(i,idx);
    %X of that movie, etc
    X_grad(i,:)=(X(i,:)*Theta_t'-Y_t)*Theta_t+lambda*X(i,:);
end
for i=1:num_users
    %Find movies that the user has rated
    idx=find(R(:,i)==1);
    %Get features for those movies
    X_t=X(idx,:);
    %For a given users, get ratings for all movies
    Y_t=Y(idx,i);
    Theta_grad(i,:)=(X_t*Theta(i,:)'-Y_t)'*X_t+lambda*Theta(i,:);
end
    














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
