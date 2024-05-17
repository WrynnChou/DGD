from main import *

if __name__ == '__main__':

    lr, num_epochs = 0.03, 1000
    batch_size = 3000

    net = LeNet5()
    train_iter, test_iter = load_data_mnist(batch_size=batch_size)
    l1, l2 = train(net, train_iter,  test_iter, train_iter, num_epochs, lr, try_gpu(), init='weights/params.pkl')

    net2 = LeNet5()
    train_iter2, test_iter2 = load_data_DDS(batch_size=batch_size, path='data/mnist_pp1.csv')
    train(net2, train_iter2, test_iter2, train_iter2, num_epochs, lr, try_gpu(), init='weights/params.pkl')

