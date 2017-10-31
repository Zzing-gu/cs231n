import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] # 500
  num_classes = W.shape[1]  # 3073
  loss = 0.0   # 구하고자 하는 값
  for i in xrange(num_train):
    # Compute vector of scores
    f_i = X[i].dot(W)   # score 구하기

    # Normalization trick to avoid numerical instability, per http://cs231n.github.io/linear-classify/#softmax
    f_i -= np.max(f_i)    # 값이 더 잘나오게 하는 트릭이라고 함

    # Compute loss (and add to it, divided later)
    sum_j = np.sum(np.exp(f_i))            # 밑변 구하기
    p = lambda k: np.exp(f_i[k]) / sum_j   # 익명의 함수를 만들어 변수 p 에 저장 ...     밑변과 윗변을 위한 식 ..  k 는 입력 값으로 yi 에 해당함
    loss += -np.log(p(y[i]))               # 방금 만든 p 함수에  yi  정답 레이블 값을 넣고 -log 후 loss 값에 누적 시키기

    # Compute gradient
    # Here we are computing the contribution to the inner sum for a given i.
    for k in range(num_classes):
      p_k = p(k)
      dW[:, k] += (p_k - (k == y[i])) * X[i]     #이해 힘듬

  loss /= num_train   # 여태껏 누적시킨 loss 값 평균 내기
  loss += 0.5 * reg * np.sum(W * W)   #  regulation 값 더하기    loss function 완성
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
