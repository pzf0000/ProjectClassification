class BaseInput:
    """
    基本输入接口
    """
    def read(self, **kwargs):
        return self.input(**kwargs)


class BaseOutput:
    """
    基本输出接口
    """
    def write(self):
        return self.output()


class BaseIO(BaseInput, BaseOutput):
    pass


class IO(BaseIO):
    pass
