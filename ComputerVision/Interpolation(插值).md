1. 什么是插值

插值就是在修改图像尺寸时，利用已知的点来“猜”未知的点，利用旧图像矩阵中的点计算新矩阵图像的点并插入

2. 常用的插值算法

  1）最近邻近：
  
      将目标图像各点的像素值设为源图像中与其最近的点。算法优点在于简单、速度快，缺点在于会破坏原图像中像素的渐变关系。
      
      代码实现：
      
          def nearest_neighbor_resize(img, new_w, new_h):
              # height and width of the input img
              h, w = img.shape[0], img.shape[1]
              
              # new image with rgb channel
              ret_img = np.zeros(shape=(new_h, new_w, 3), dtype='uint8')
              
              # scale factor
              s_h, s_c = (h * 1.0) / new_h, (w * 1.0) / new_w

              # insert pixel to the new img
              for i in xrange(new_h):
                  for j in xrange(new_w):
                      p_x = int(j * s_c)
                      p_y = int(i * s_h)

                      ret_img[i, j] = img[p_y, p_x]

              return ret_img
  
  2）单线性插值
  
      求两点之间的直线方程，通过直线方程求出插入点的值。
      
      (y1-y0)/(x1-x0) = (y-y0)/(x-x0),  y = (x1-x)/(x1-x0) * y1 + (x-x0)/(x1-x0) * y0
  
  3）双线性插值
  
      先在x方向使用单线性插值，再在y方向使用单线性插值
  
  4）双三次插值
