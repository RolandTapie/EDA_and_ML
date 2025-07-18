from IPython.terminal.magics import get_pasted_lines


class ProjectContext():

    def __init__(self):
        self.title=""
        self.reason=""
        self.goal=""
        self.source=""
        self.context=""

    def add_project_title(self,title):
        self.title= title
        self.context = "Project title :" + title +"\n"
        return self

    def add_project_reason(self,reason):
        self.reason = reason
        self.context = self.context + "\n" +"Project Reason :"+ reason + "\n"
        return self

    def add_project_goal(self, goal):
        self.goal = goal
        self.context = self.context + "\n" +"Project Goal :"+ goal + "\n"
        return self

    def add_project_data_source (self, source):
        self.source = source
        self.context = self.context + "\n" +"Project Data source :"+ source + "\n"
        return self

    def get_context(self):
        return self.context



