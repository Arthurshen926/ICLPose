import torch
from feature_extract.tools.vfm.mapping_joint_coordinate_loss import joint_error


def fixture():
    dtype=torch.float64
    return [torch.tensor(x,dtype=dtype) for x in [
        [[0,0]],[[.2,-.1]],[[0,0]],[[.1,.1]],[[1,2,5]],
        [[[1,0],[0,1],[0,0]]],[[[100,0,128],[0,100,72],[0,0,1]]],[.01],[[-.25,-.25]],[[.25,.25]]]]


def test_ideal_coordinates_have_zero_joint_error_on_nonzero_height_anchor():
    values=fixture();values[0]=values[1].clone();values[2]=values[3].clone()
    torch.testing.assert_close(joint_error(*values),torch.zeros((1,2),dtype=torch.float64))


def test_joint_coordinate_loss_has_correct_gradients():
    values=fixture();values[0].requires_grad_();values[2].requires_grad_()
    assert torch.autograd.gradcheck(lambda image,uv:joint_error(image,values[1],uv,*values[3:]),(values[0],values[2]))
