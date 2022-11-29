from interface import Args, Interface

def run():
    args = Args()
    # args = _Args()
    interf = Interface(args())
    interf.run()




if __name__ == '__main__':
    run()

    

