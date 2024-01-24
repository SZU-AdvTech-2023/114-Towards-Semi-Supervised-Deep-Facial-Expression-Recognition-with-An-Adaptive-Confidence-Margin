class ConsoleLogger:
    def __init__(self, log_console_path, console_content, title=None):
        self.console_content = console_content
        self.log_console_path = log_console_path
        self.title = '' if title is None else title
        self.log_console = open(self.log_console_path, 'a', encoding='utf-8')

    def append(self, new_console_content):
        self.log_console.write(new_console_content)
        self.log_console.write('\n')
        self.log_console.flush()