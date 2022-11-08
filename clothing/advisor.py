from meta import *



class Advisor(MetaModule):
    def __init__(self, input, hidden1, output):
        super(Advisor, self).__init__()
        self.linear_feature = MetaLinear(input, hidden1)
        self.linear_loss = MetaLinear(1, hidden1)

        self.relu = nn.ReLU()
        self.linear_f_out = MetaLinear(hidden1 * 2, output)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):

        x = self.linear_feature(x)
        x = self.relu(x)

        y = self.linear_loss(y)
        y = self.relu(y)

        conc_f = torch.cat([x, y], dim=1)

        out_f = self.linear_f_out(conc_f)
        out_f = self.sigmoid(out_f)

        return out_f
