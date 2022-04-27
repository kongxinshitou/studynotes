# Goroutine笔记

### 1.main函数和其他goroutine的关系

>在Go里每一个并发执行的活动称为goroutine，当一个程序启动时，只有一个goroutine来调用main函数，称它为主goroutine。新的goroutine通过go语句进行创建。当main函数返回，当它发生时，所有的goroutine都暴力地直接终结。[^实践与理解]

[^实践与理解]: 如果没有约束，那么main函数的goroutine与main函数里面的goroutine将会异步地执行，这样的话可能goroutine还没有执行完成就被强制退出了，这时候需要使用一些方法，比如使用sync包里面的WaitGroup。

```go
//定义一个计数器来记录并维护运行的goroutine
var wg sync.WaitGroup
...
//计数器数量加2
wg.Add(2)
...
//使用defer来使得无论程序是否正常结束都能顺利执行wg.Done(),其中wg.Done()等价于wg.Add(-1)
defer wg.Done()
...
//wg.wait()会等待计数器清零。可以通过在主程序中设置wg.Wait来保证main函数不会在goroutine执行
//完就终止。
wg.Wait()

```



