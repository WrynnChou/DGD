from main import *

if __name__ == '__main__':

    lr, num_epochs = 0.05, 600
    batch_size = 2500
    net = MLP(768, 10)
    net2 = MLP(768, 10)
    net3 = MLP(768, 10)
    net4 = MLP(768, 10)
    
    train_iter, test_iter = load_data_cifar(batch_size=batch_size, path='data/cifar_feature.txt', path_t='data/cifar_test.csv')
    train(net, train_iter, test_iter, train_iter, num_epochs, lr, try_gpu())

    train_iter2, test_iter2 = load_data_cifar2(batch_size=batch_size, path='data/cifar_rearranged_2500.csv', path_t='data/cifar_test.csv')
    train(net2, train_iter2, test_iter2, train_iter, num_epochs, lr, try_gpu())

    # install GraB first
    # train_grab(net3, 'data/cifar_feature.txt', 'data/cifar_test.csv', batch_size, num_epochs, lr, try_gpu())

    train_iter4, test_iter4 = load_data_cifar4(batch_size=batch_size, path='data/cifar_feature.txt', path_t='data/cifar_test.csv')
    train(net, train_iter4, test_iter4, train_iter4, num_epochs, lr, try_gpu())

#     train_iter, test_iter = load_data_cifar(batch_size=batch_size, path='data/feature_flower.txt', path_t='data/flower_test.csv')
#     train(net, train_iter, test_iter, train_iter, num_epochs, lr, try_gpu())
#
#     train_iter, test_iter = load_data_cifar2(batch_size=batch_size, path='data/feature_flower_250.csv', path_t='data/flower_test.csv')
#     train(net2, train_iter, test_iter, train_iter, num_epochs, lr, try_gpu())

#     train_grab(net3, 'data/feature_flower.txt', 'data/flower_test.csv', batch_size, num_epochs, lr, try_gpu())

#     train_iter4, test_iter4 = load_data_cifar4(batch_size=batch_size, path='data/feature_flower.txt', path_t='data/flower_test.csv')
#     train(net4, train_iter4, test_iter4, train_iter4, num_epochs, lr, try_gpu())
