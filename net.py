from __future__ import print_function
import os
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import numpy as np
import os.path as osp
import os
from warnings import warn
import pickle
import config as cfgs

from decompose import VH_decompose, Network_decouple, ITQ_decompose, kernel_svd
from builder import Net as NetBuilder
from utils import underline, OK, shell, Timer

class Net():
    def __init__(self, pt, model=None, phase = caffe.TEST, accname=None, mask_layers=None,\
        SD_param=None, ND_param=None, CD_param=None, gpu=None, nSamples=None, nPointsPerSample=None, frozen_name=None):

        # self.caffe_device()
        if gpu is None:
            gpu = cfgs.device_id
        assert gpu >= -1
        if gpu != -1:
            caffe.set_mode_gpu()
            caffe.set_device(gpu)
            print("using GPU ID %d" % gpu)
        else:
            # caffe.set_mode_cpu()
            print("using CPU caffe")
        self.net = caffe.Net(pt, phase)#, level=2) # creates net but not load weights -by Mario
        self.pt_dir = pt
        if model is not None:
            self.net.copy_from(model)
            self.caffemodel_dir = model
        else:
            self.caffemodel_dir = pt.replace('prototxt','caffemodel')
        self._protocol = 4
        self.net_param = NetBuilder(pt=pt) # instantiate the NetBuilder -by Mario
        self.num = self.blobs_num('data')
        self._layers = dict()
        self._bottom_names = None
        self._top_names = None
        if cfgs.data_layer is not None:
            self.data_layer = cfgs.data_layer
        else:
            self.data_layer = 'data'
        if self.get_layer(self.data_layer).type=='MemoryData':
            # to define if use set_input_arrays to get data from memory
            self._mem = True
        else:
            self._mem = False

        self._accname = accname if accname is not None else cfgs.accname
        self.mask_layers = mask_layers if mask_layers is not None else cfgs.mask_layers
        self.ND_param = ND_param if ND_param is not None else cfgs.ND_param
        self.SD_param = SD_param if SD_param is not None else cfgs.SD_param
        self.CD_param = CD_param if CD_param is not None else cfgs.CD_param
        self.nSamples = nSamples if nSamples is not None else cfgs.nSamples
        self.nPointsPerSample = nPointsPerSample if nPointsPerSample is not None else cfgs.nPointsPerSample
        self.frozen_name = frozen_name if frozen_name is not None else cfgs.frozen_name

        self.resnet = False # 'res' in self.pt_dir or 'Res' in self.pt_dir
        self.acc=[]

        self.WPQ={} # stores pruned values, which will be saved to caffemodel later (since Net structure couldn't be dynamically changed in caffe)

        self.convs= self.type2names()  # convs contains a list of strings -by Mario
        self.spation_convs = self.type2names('ConvolutionDepthwise')
        for c in self.convs:
            if self.conv_param(c).group != 1:
                self.spation_convs.append(c)
        self.relus = self.type2names(layer_type='ReLU')
        self.bns = self.type2names(layer_type='BatchNorm')
        self.affines = self.type2names(layer_type='Scale')
        self.pools = self.type2names(layer_type='Pooling')
        self.sums = self.type2names('Eltwise')
        self.concats = self.type2names('Concat')
        self.innerproduct = self.type2names('InnerProduct')

    def caffe_device(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfgs.device_id)

    @property
    def _frozen(self):
        if self.frozen_name is None:
            frozenname = 'frozen' + str(self.nSamples)
        else:
            frozenname = self.frozen_name
        return osp.join(osp.split(self.pt_dir)[0], frozenname+".pickle")

    @property
    def top_names(self):
        if self._top_names is None:
            self._top_names = self.net.top_names
        return self._top_names

    @property
    def bottom_names(self):
        if self._bottom_names is None:
            self._bottom_names = self.net.bottom_names
        return self._bottom_names

    def layer_bottom(self, name):
        return self.net_param.layer_bottom(name)

    def _save(self, new_name, orig_name, prefix='acc'):
        if new_name is None:
            # avoid overwrite
            path, name = osp.split(orig_name)
            new_name = osp.join(path, underline(prefix, name))
        else:
            print("overwriting", new_name)
        return new_name


    def save_pt(self, new_name=None, **kwargs):
        new_name = self._save(new_name, self.pt_dir, **kwargs)
        self.net_param.write(new_name)
        return new_name

    def save_caffemodel(self, new_name=None, **kwargs):
        new_name = self._save(new_name, self.caffemodel_dir, **kwargs)
        self.net.save(new_name)
        return new_name

    def save(self, new_pt=None, new_caffemodel=None, **kwargs):
        return self.save_pt(new_pt, **kwargs), self.save_caffemodel(new_caffemodel, **kwargs)

    def param(self, name):
        if name in self.net.params:
            return self.net.params[name]
        else:
            raise Exception("no this layer: %s" % name)

    def blobs(self, name):
        return self.net.blobs[name]

    def forward(self, test=True):
        ret = self.net.forward()
        self.cum_acc(ret)
        return ret

    def param_w(self, name):
        return self.param(name)[0]

    def param_b(self, name):
        return self.param(name)[1]

    def param_data(self, name):
        return self.param_w(name).data

    def param_b_data(self, name):
        return self.param_b(name).data

    def set_param_data(self, name, data):
        if isinstance(name, tuple):
            self.param(name[0])[name[1]].data[...] = data.copy()
        else:
            self.param_w(name).data[...] = data.copy()

    def set_param_b(self, name, data):
        self.param_b(name).data[...] = data.copy()

    def ch_param_data(self, name, data):
        if isinstance(name, tuple):
            if name[1] == 0:
                self.ch_param_data(name[0], data)
            elif name[1] == 1:
                self.ch_param_b(name[0], data)
            else:
                NotImplementedError
        else:
            self.param_reshape(name, data.shape)
            self.param_w(name).data[...] = data.copy()

    def ch_param_b(self, name, data):
        self.param_b_reshape(name, data.shape)
        self.param_b(name).data[...] = data.copy()

    def param_shape(self, name):
        return self.param_data(name).shape

    def param_b_shape(self, name):
        return self.param_b_data(name).shape

    def param_reshape(self, name, shape):
        self.param_w(name).reshape(*shape)

    def param_b_reshape(self, name, shape):
        self.param_b(name).reshape(*shape)

    def data(self, name='data', **kwargs):
        return self.blobs_data(name, **kwargs)

    def label(self, name='label', **kwargs):
        return self.blobs_data(name, **kwargs)


    def blobs_data(self, name, **kwargs):
        return self.blobs(name, **kwargs).data

    def blobs_type(self, name):
        return self.blobs_data(name).dtype

    def blobs_shape(self, name):
        return self.blobs_data(name).shape

    def blobs_reshape(self, name, shape):
        return self.blobs(name).reshape(*shape)

    def blobs_num(self, name):
        return self.blobs(name).num

    def blobs_count(self, name):
        return self.blobs(name).count

    def blobs_height(self, name):
        return self.blobs(name).height
    def blobs_channels(self, name):
        return self.blobs(name).channels

    def blobs_width(self, name):
        return self.blobs(name).width

    def blobs_CHW(self, name):
        return self.blobs_count(name) / self.blobs_num(name)

    # =============== protobuf ===============

    def get_layer(self, conv):
        """return self.net_param.layer[conv][0]"""
        return self.net_param.layer[conv][0]

    def conv_param_stride(self, conv):
        stride = self.conv_param(conv).stride
        if len(stride) == 0:
            return 1
        else:
            assert len(stride) == 1
            return stride[0]

    def conv_param_pad(self, conv):
        pad = self.conv_param(conv).pad
        assert len(pad) == 1
        return pad[0]

    def conv_param_kernel_size(self, conv):
        kernel_size = self.conv_param(conv).kernel_size
        assert len(kernel_size) == 1
        return kernel_size[0]

    def conv_param_num_output(self, conv):
        return self.conv_param(conv).num_output

    def net_param_layer(self, conv):
        """return self.net_param.layer[conv]"""
        return self.net_param.layer[conv]

    def conv_param(self, conv):
        return self.get_layer(conv).convolution_param

    def corr_param(self, conv):
        return self.get_layer(conv).correlation_param

    def set_conv(self, conv, num_output=0, new_name=None, pad_h=None, pad_w=None, kernel_h=None, kernel_w=None, stride=None, bias=None,group=None):
        conv_param = self.conv_param(conv)
        if num_output != 0:
            conv_param.num_output = type(conv_param.num_output)(num_output)
        if pad_h is not None:
            while len(conv_param.pad):
                conv_param.pad.remove(conv_param.pad[0])
            conv_param.pad.append(pad_h)
            conv_param.pad.append(pad_w)
        if kernel_h is not None:
            while len(conv_param.kernel_size):
                conv_param.kernel_size.remove(conv_param.kernel_size[0])
            conv_param.kernel_size.append(kernel_h)
            conv_param.kernel_size.append(kernel_w)

        if stride is not None:
            while len(conv_param.stride):
                conv_param.stride.remove(conv_param.stride[0])
            for i in stride:
                conv_param.stride.append(i)

        if bias is not None:
            conv_param.bias_term = bias

        if group is not None:
            conv_param.group = group

        if new_name is not None:
            self.net_param.ch_name(conv, new_name)

    # =============== data ===============

    def extract_features(self, names=[], nBatches=None, points_dict=None, save=False):
        assert nBatches is None, "deprecate"
        nBatches = self.nSamples
        nPointsPerLayer=self.nPointsPerSample
        if not isinstance(names, list):
            names = [names]

        DEBUG = False

        pads = dict()
        shapes = dict()
        feats_dict = dict()
        def set_points_dict(name, data):
            assert name not in points_dict
            points_dict[name] = data
        
        if save:
            if points_dict is None:
                frozen_points = False
                points_dict = dict()

                set_points_dict("nPointsPerLayer", nPointsPerLayer)
                set_points_dict("nBatches", nBatches)
            else:
                frozen_points = True
                if nPointsPerLayer != points_dict["nPointsPerLayer"] or nBatches != points_dict["nBatches"]:
                    print("overwriting nPointsPerLayer, nBatches with frozen_points")

                nPointsPerLayer = points_dict["nPointsPerLayer"]
                nBatches = points_dict["nBatches"]

        assert len(names) > 0

        nPicsPerBatch = self.blobs_num(names[0])
        nFeatsPerBatch = nPointsPerLayer  * nPicsPerBatch
        print("run for", nBatches, "batches", "nFeatsPerBatch", nFeatsPerBatch)
        nFeats = nFeatsPerBatch * nBatches

        for name in names:
            """avoiding X out of bound"""
            shapes[name] = (self.blobs_height(name), self.blobs_width(name))
            feats_dict[name] = np.ndarray(shape=(nFeats, self.blobs_channels(name)))      # feat_dict is (50000,channels)
            print("Extracting", name, feats_dict[name].shape)
        
        idx = 0
        if save:
            if not frozen_points:
                set_points_dict("data", self.data().shape)
                set_points_dict("label", self.label().shape)

        runforn = self.nSamples
        for batch in range(runforn):
            if save:
                if not frozen_points:
                    self.forward()
                    set_points_dict((batch, 0), self.data().copy())
                    set_points_dict((batch, 1), self.label().copy())
                else:
                    self.net.set_input_arrays(points_dict[(batch, 0)], points_dict[(batch, 1)])
                    self.forward()
            else:
                self.forward()

            for name in names:
                # pad = pads[name]
                shape = shapes[name]
                feat = self.blobs_data(name)
                if DEBUG: print(name, self.blobs_shape(name))
                # TODO!!! different patch for different image per batch
                if save:
                    if not frozen_points or (batch, name, "randx") not in points_dict:
                        #embed()
                        randx = np.random.randint(0, shape[0]-0, nPointsPerLayer)
                        randy = np.random.randint(0, shape[1]-0, nPointsPerLayer)
                        """
                        TODO: optimize the data-driven SD in res-block or dense block structure
                        by using sum output for feature reconstruction
                        """
                        
                        if self.resnet:
                            branchrandxy = None
                            branch1name = '_branch1'
                            branch2cname = '_branch2b'
                            if name in self.sums:
                                # the previous sum and branch2b will be identical
                                branchrandxy = name + branch2cname
                            
                            if branch2cname in name:
                                branchrandxy = name.replace(branch2cname, branch1name)
                                if branchrandxy not in self.convs:
                                    branchrandxy = None

                            if branchrandxy is not None:
                                if 0: print('pointsdict of', branchrandxy, 'identical with', name)
                                randx = points_dict[(batch, branchrandxy , "randx")]
                                randy = points_dict[(batch, branchrandxy , "randy")]
                        
                        set_points_dict((batch, name, "randx"), randx.copy())
                        set_points_dict((batch, name, "randy"), randy.copy())

                    else:
                        if name not in self.sums:
                            randx = points_dict[(batch, name, "randx")]
                            randy = points_dict[(batch, name, "randy")]
                        else:
                            # extract the same position as next sum groud truth on each feature map
                            next_sum = self.sums[self.sums.index(name)+1]
                            randx = points_dict[(batch, next_sum, "randx")]
                            randy = points_dict[(batch, next_sum, "randy")]
                else:
                    randx = np.random.randint(0, shape[0]-0, nPointsPerLayer)
                    randy = np.random.randint(0, shape[1]-0, nPointsPerLayer)

                for point, x, y in zip(range(nPointsPerLayer), randx, randy):

                    i_from = idx+point*nPicsPerBatch
                    try:
                        feats_dict[name][i_from:(i_from + nPicsPerBatch)] = feat[:,:,x, y].reshape((self.num, -1))
                    except:
                         print('total', runforn, 'batch', batch, 'from', i_from, 'to', i_from + nPicsPerBatch)
                         raise Exception("out of bound")

            idx += nFeatsPerBatch

        self.clr_acc()
        if save:
            if frozen_points:
                if points_dict is not None:
                    return feats_dict, points_dict
                return feats_dict
            else:
                return feats_dict, points_dict
        else:
            return feats_dict

    def extract_XY(self, X, Y, DEBUG = False, w1=None):

        """
        given two conv layers, extract X (n, c0, ks, ks), given extracted Y(n, c1, 1, 1)
        NOTE only support: conv(X) relu conv(Y)

        Return:
            X feats of size: N C h w
        """
        pad = self.conv_param_pad(Y)
        kernel_size = self.conv_param_kernel_size(Y)
        half_kernel_size = int(kernel_size/2)
        if w1 is not None:
            gw1=True
            x_pad = self.conv_param_pad(w1)
            x_ks = self.conv_param_kernel_size(w1)
            x_hks = int(x_ks/2)
        else:
            # assert x_hks == 0
            gw1=False
        stride = self.conv_param_stride(Y)

        print("Extracting X", X, "From Y", Y, 'stride', stride)

        X_h = self.blobs_height(X)
        X_w = self.blobs_width(X)
        Y_h = self.blobs_height(Y)
        Y_w = self.blobs_width(Y)

        def top2bottom(x, padded=1):
            """
            top x to bottom x0
            NOTE assume feature map padded
            """
            if padded:
                if gw1:
                    return half_kernel_size + x_hks + stride * x
                return half_kernel_size + stride * x
            return half_kernel_size - pad + stride * x

        """avoiding X out of bound"""
        # Y_pad = 0
        # while top2bottom(Y_pad) - half_kernel_size < 0:
        #     Y_pad += 1

        def y2x(y, **kwargs):
            """
            top y to bottom x patch
            """
            x0 = top2bottom(y, **kwargs)

            # return range(x0 - half_kernel_size, x0 + half_kernel_size + 1)
            if gw1:
                return x0 - half_kernel_size - x_hks, x0 + half_kernel_size + x_hks + 1
            return x0 - half_kernel_size, x0 + half_kernel_size + 1

        def bottom2top(x0):
            NotImplementedError
            return (x0 - half_kernel_size + pad) / stride


        shape = (Y_h, Y_w)

        idx = 0

        nPointsPerLayer = self._points_dict["nPointsPerLayer"]
        nBatches = self._points_dict["nBatches"]
        if gw1:
            nPicsPerBatch = self.blobs_num(X)
            nFeatsPerBatch = nPointsPerLayer  * nPicsPerBatch
            nFeats = nFeatsPerBatch * nBatches
            feat_h = (kernel_size+x_ks) - 1
            assert kernel_size % 2 == 1
            assert x_ks % 2 == 1
            feats_dict = np.ndarray(shape=(nFeats, self.blobs_channels(X), feat_h ,feat_h ))
            pass
        else:
            nPicsPerBatch = self.blobs_num(X) * kernel_size * kernel_size
            nFeatsPerBatch = nPointsPerLayer  * nPicsPerBatch
            nFeats = nFeatsPerBatch * nBatches

            feats_dict = np.ndarray(shape=(nFeats, self.blobs_channels(X)))
        X_shape = self.blobs_shape(X)

        feat_pad = pad if not gw1 else pad + x_pad
        
        for batch in range(nBatches):
            if DEBUG: print("done", batch, '/', nBatches)

            if self._mem:
                self.net.set_input_arrays(self._points_dict[(batch, 0)], self._points_dict[(batch, 1)])

            self.forward()

            # padding

            feat = np.zeros((X_shape[0], X_shape[1], X_shape[2] + 2 * feat_pad, X_shape[3] + 2 * feat_pad), dtype=self.blobs_type(X))
            feat[:, :, feat_pad:X_shape[2] + feat_pad, feat_pad:X_shape[3] + feat_pad] = self.blobs_data(X).copy()

            randx = self._points_dict[(batch, Y, "randx")]
            randy = self._points_dict[(batch, Y, "randy")]

            for point, x, y in zip(range(nPointsPerLayer), randx, randy):

                i_from = idx+point*nPicsPerBatch
                """n hwc"""

                x_start, x_end = y2x(x)
                y_start, y_end = y2x(y)
                if gw1:
                    if 0:
                        try:
                            feats_dict[i_from:(i_from + nPicsPerBatch)] = \
                            feat[:, :, x_start:x_end, y_start:y_end].copy()
                        except:
                            embed()
                    else:
                        feats_dict[i_from:(i_from + nPicsPerBatch)] = \
                        feat[:, :, x_start:x_end, y_start:y_end].copy()

                else:
                    feats_dict[i_from:(i_from + nPicsPerBatch)] = \
                    np.moveaxis(feat[:, :, x_start:x_end, y_start:y_end], 1, -1).reshape((nPicsPerBatch, -1))

            if DEBUG:
                # sanity check using relu(WX + B) = Y
                # embed()
                # # W: chw n
                # W = self.param_data(X).reshape(self.param_shape(X)[0], -1).T
                # # b
                # b = self.param_b_data(X)
                # n hwc
                bottom_X = feat[:, :, x_start:x_end, y_start:y_end].reshape((self.num, -1))

                # W2: chw n
                W2 = self.param_data(Y).reshape(self.param_shape(Y)[0], -1).T
                # b2
                b2 = self.param_b_data(Y)

                fake = relu(bottom_X).dot(W2) + b2

                # n c
                real = self.blobs_data(Y)[:,:,x, y].reshape((self.num, -1))

                CHECK_EQ(fake, real)

            idx += nFeatsPerBatch

        self.clr_acc()
        return feats_dict

    def freeze_images(self, check_exist=False, convs=None, **kwargs):
        frozen = self._frozen
        if check_exist:
            if osp.exists(frozen):
                print("Exists", frozen)
                return frozen

        if convs is None:
            convs = self.type2names()
        feats_dict, points_dict = self.extract_features(names=convs, save=1, **kwargs)

        data_layer = self.data_layer
        if len(self.net_param_layer(data_layer)) == 2:
            self.net_param.net.layer.remove(self.get_layer(data_layer))

        i = self.get_layer(data_layer)
        i.type = "MemoryData"
        i.memory_data_param.batch_size = points_dict['data'][0]
        i.memory_data_param.channels = points_dict['data'][1]
        i.memory_data_param.height = points_dict['data'][2]
        i.memory_data_param.width = points_dict['data'][3]
        i.ClearField("transform_param")
        i.ClearField("data_param")
        i.ClearField("include")

        print("wrote memory data layer to", self.save_pt(prefix="mem"))
        print("freezing imgs to", frozen)
        with open(frozen, 'wb') as f:
            pickle.dump([feats_dict, points_dict], f, protocol=self._protocol)

        return frozen

    def dis_memory(self):
        data_layer = self.data_layer
        i = self.get_layer(data_layer)
        i.ClearField("memory_data_param")
        i.type = "Data"
        if cfgs.dataset=='imagenet':
            i.transform_param.crop_size = 224
            i.transform_param.mirror = False
            if len(i.transform_param.mean_value) == 0:
                i.transform_param.mean_value.extend([104.0,117.0,123.0])
            i.data_param.source = cfgs.imagenet_val
            i.data_param.batch_size = 20
            i.data_param.backend = i.data_param.LMDB
        elif cfgs.dataset=='cifar10':
            i.transform_param.scale = .0078125
            if len(i.transform_param.mean_value) == 0:
                i.transform_param.mean_value.extend([128])
            i.transform_param.mirror = False
            i.data_param.source = cfgs.cifar10_val
            i.data_param.batch_size = 128
            i.data_param.backend = i.data_param.LMDB

        else:
            assert False

        ReLUs = self.type2names("ReLU")
        Convs = self.type2names()

        # merge back relus
        for r in ReLUs:
            if self.top_names[r][0] == r:
                assert len(self.bottom_names[r]) == 1
                conv = self.bottom_names[r][0]
                self.net_param.ch_top(r, conv, r)
                for i in self.net_param.layer:
                    if i != r:
                        self.net_param.ch_bottom(i, conv, r)

    def load_frozen(self, DEBUG=False, feats_dict=None, points_dict=None):
        if feats_dict is not None:
            print("loading imgs from memory")
            self._feats_dict = feats_dict
            self._points_dict = points_dict
            return

        frozen = self._frozen
        print("loading imgs from", frozen)
        with open(frozen, 'rb') as f:
            self._feats_dict, self._points_dict = pickle.load(f)

        if DEBUG:
            convs = self.type2names()
            feats_dict = self.extract_features(convs, points_dict=self._points_dict, save=1)
            print("feats_dict", feats_dict)
            print("self._feats_dict", self._feats_dict)
            embed()
            for i in feats_dict:
                for x, y in zip(np.nditer(self._feats_dict[i]), np.nditer(feats_dict[i])):
                    assert  x == y
            OK("frozen         ")

        print("loaded")

    def type2names(self, layer_type='Convolution'):
        if layer_type not in self._layers:
            self._layers[layer_type] = self.net_param.type2names(layer_type)
        return self._layers[layer_type]


    def insert(self, bottom, name=None, layer_type="Convolution", bringforward=True, update_nodes=None, bringto=None, DP=False, next_layer=None,**kwargs):
        if layer_type=="Convolution":
            # insert
            self.net_param.set_cur(bottom)
            self.net_param.Convolution(name, bottom=[bottom], **kwargs)
            # clone previous layer
            if "stride" not in kwargs:
                new_conv_param = self.conv_param(name)
                while len(new_conv_param.stride):
                    new_conv_param.stride.remove(new_conv_param.stride[0])
                for i in self.conv_param(bottom).stride:
                    new_conv_param.stride.append(i)

            # update input nodes for others
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                if self.net_param.layer[i][0].type == 'Eltwise' and self.net_param.layer[i][0].name == next_layer and DP:
                    # for eltwise summation, add one bottom instead of change
                    self.net_param.layer[i][0].bottom.extend([name])
                    self.net_param.layer[i][0].eltwise_param.coeff.extend([1.0])
                elif next_layer is None:
                    self.net_param.ch_bottom(i, name, bottom)

            # for i, bot in self.bottom_names.items():
            #     if bottom in bot:
            #         assert len(bot) == 1, "only support single pass"
            #         self.net_param.ch_bottom(i, name, bottom)
            if bringforward:
                if bringto is not None:
                    bottom = bringto
                self.net_param.bringforward(bottom)
        
        elif layer_type == "ConvolutionDepthwise":
            # insert
            self.net_param.set_cur(bottom)
            self.net_param.ConvolutionDepthwise(name, bottom=[bottom], **kwargs)

            # update input nodes for others
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                if self.net_param.layer[i][0].type == 'Eltwise' and self.net_param.layer[i][0].name == next_layer and not DP:
                    # for eltwise summation, add one bottom instead of change
                    self.net_param.layer[i][0].bottom.extend([name])
                    self.net_param.layer[i][0].eltwise_param.coeff.extend([1.0])
                elif next_layer is None:
                    self.net_param.ch_bottom(i, name, bottom)

            if bringforward:
                if bringto is not None:
                    bottom = bringto
                self.net_param.bringforward(bottom)

        elif layer_type == "Eltwise":
            # insert
            self.net_param.set_cur(None)
            self.net_param.Eltwise(name, bottom, no_bottom=True)
            if update_nodes is None:
                update_nodes = self.net_param.layer
            for i in update_nodes:
                if i == name:
                    continue
                self.net_param.ch_bottom(i, name, bottom)
            if bringforward:
                if bringto is not None:
                    bottom = bringto
                self.net_param.bringforward(bottom)

    def remove(self, name, inplace=False):
        self.net_param.rm_layer(name, inplace)

    def accuracy(self):
        times = self.nSamples
        acc = []
        for i in range(times):
            res = self.forward()
            acc.append(float(res[self._accname]))

        return np.mean(acc) #, 'std', np.std(acc)

    def cum_acc(self, res):
        self.acc.append(float(res[self._accname]))

    def clr_acc(self, show=True):
        self.currentacc = np.mean(self.acc)
        if show:
            print('Acc {:7.3f}'.format(self.currentacc*100))
        self.acc = []

    def finalmodel(self, WPQ=None, **kwargs): # the prefix for the name of the saved model is added by self.linear() -by Mario
        """ load weights into caffemodel"""
        if WPQ is None:
            WPQ = self.WPQ
        return self.linear(WPQ, **kwargs)

    def infer_pad_kernel(self, W, origin_name, conv_param=None):
        num_output, _, kernel_h, kernel_w = W.shape
        # only support for odd kernel_size
        assert kernel_h%2 == 1
        assert kernel_w%2 == 1
        pad_h = int(kernel_h/2)
        pad_w = int(kernel_w/2)
        if origin_name is not None:
            stride = self.conv_param(origin_name).stride
        elif conv_param is not None:
            stride = conv_param.stride
        else:
            NotImplementedError
        if len(stride) == 1:
            pass
        elif len(stride) == 0:
            stride = [1]
        else:
            NotImplementedError
        if stride[0] == 1:
            pass
        elif stride[0] >= 2:
            stride = [stride[0] if pad_h else 1, stride[0] if pad_w else 1]
            # stride = [1 if pad_h else stride[0], 1 if pad_w else stride[0]]
            warn("stride larger than 1 decompose dangerous")
        else:
            NotImplementedError
        return {"pad_h":pad_h, "pad_w":pad_w, "kernel_h":kernel_h, "kernel_w":kernel_w, "num_output":num_output, "stride":stride}
    
    # =========algorithms=========

    def linear(self, WPQ, prefix='decomp', save=True,DEBUG=0):
        for i, j in WPQ.items():
            if save:
                self.set_param_data(i, j)
            else:
                self.ch_param_data(i, j)
        if save:
            return self.save_caffemodel(prefix=prefix)

    def layercomputation(self, conv, channels=1., outputs=1.):
        bottom = self.bottom_names[conv]
        assert len(bottom) == 1
        bottom = bottom[0]
        s = self.blobs_shape(bottom)
        if conv in self.type2names('Correlation'):
            param = self.corr_param(conv)
            patch = (param.displacement*2+1)**2
            channels *= s[1]
            c = s[2]*s[3]*channels*patch
        else:
            p = self.param_shape(conv)
            if conv in self.convs and conv not in self.spation_convs:
                channels *= p[1]
                outputs *= p[0]
                c = s[2]*s[3]*outputs*channels*p[2]*p[3] / self.conv_param_stride(conv)**2
            elif conv in self.spation_convs:
                if conv in self.type2names('ConvolutionDepthwise'):
                    c = s[2]*s[3]*p[0]*p[2]*p[3] / self.conv_param_stride(conv)**2
                else:
                    group = self.conv_param(conv).group
                    assert p[0]%group==0
                    outputs *= p[0]
                    channels *= p[1]/group
                    c = s[2]*s[3]*outputs*channels*p[2]*p[3] / self.conv_param_stride(conv)**2
            elif conv in self.innerproduct:
                c = p[0]*p[1]
            else:
                pass
        return int(c)

    def computation(self, params=False):
        comp=0
        if params:
            NotImplementedError
        else:
            l = []
            layers = self.convs + self.spation_convs + self.innerproduct + self.type2names('Correlation')
            for conv in layers:
                l.append(self.layercomputation(conv))
        comp = sum(l)
        for conv,i in zip(layers, l):
            print(conv, i, float(i*1000./comp))
        print("flops", comp)
        return comp

    def getBNaff(self, bn, affine, scale=1.):
        bn_layer = self.net_param.layer[bn][0]
        eps = bn_layer.batch_norm_param.eps
        mean = scale * self.param_data(bn)
        variance = (scale * self.param_b_data(bn) + eps)**.5
        k =  self.param_data(affine)
        b =  self.param_b_data(affine)
        return mean, variance, k, b

    def merge_bn(self, DEBUG=0):
        """
        Return:
            merged Weights
        """
        nobias=False
        def scale2tensor(s, weights):
            naxis = len(weights.shape)
            flatten_shape = [len(s)] + [1 for i in range(naxis-1)]
            return s.reshape(flatten_shape)

        BNs = self.type2names("BatchNorm")
        Affines = self.type2names("Scale")
        ReLUs = self.type2names("ReLU")
        Convs = self.type2names() + self.type2names("InnerProduct") + self.type2names("ConvolutionDepthwise")
        assert len(BNs) == len(Affines)

        WPQ = dict()
        for affine in Affines:
            # if 'loc' in affine or 'conf' in affine:
            #         continue
            if self.bottom_names[affine][0] in BNs:
                # non inplace BN
                noninplace = True
                bn = self.bottom_names[affine][0]
                conv = self.bottom_names[bn][0]
                # when last conv is connected to multiple layers, ignore this merging
                if conv not in Convs or len(self.top_names[conv]) > 1:
                    continue
                assert conv in Convs
            else:
                noninplace = False
                conv = self.bottom_names[affine][0]
                if conv not in Convs or len(self.top_names[conv]) > 1:
                    continue
                for bn in BNs:
                    if self.bottom_names[bn][0] == conv:
                        break

            triplet = (conv, bn, affine)
            print("Merging", triplet)

            if not DEBUG:
                scale = 1.

                mva = self.param(bn)[2].data[0]
                if mva != scale:
                    #raise Exception("Using moving average "+str(mva)+" NotImplemented")
                    scale /= mva

                mean, variance, k, b = self.getBNaff(bn, affine, scale)
                # y = wx + b
                # (y - mean) / var * k + b
                weights = self.param_data(conv)
                weights = weights / scale2tensor(variance, weights) * scale2tensor(k, weights)

                if len(self.param(conv)) == 1:
                    bias = np.zeros(weights.shape[0])
                    self.set_conv(conv, bias=True)
                    self.param(conv).append(self.param_b(bn))
                    nobias=True
                else:
                    bias = self.param_b_data(conv)
                bias -= mean
                bias = bias / variance * k + b

                WPQ[(conv, 0)] = weights
                WPQ[(conv, 1)] = bias

            self.remove(affine)
            self.remove(bn)
            if not noninplace:
                have_relu=False
                for r in ReLUs:
                    if self.bottom_names[r][0] == conv:
                        have_relu=True
                        break
                if have_relu:
                    self.net_param.ch_top(r, r, conv)
                    for i in self.net_param.layer:
                        if i != r:
                            self.net_param.ch_bottom(i, r, conv)

        new_pt, new_model = self.save(prefix='merged')
        return WPQ, new_pt, new_model

    def save_no_bn(self, WPQ, prefix='bn_merge'):
        self.forward()
        for i, j in WPQ.items():
            self.set_param_data(i, j)

        return self.save_caffemodel(prefix=prefix)

    def seperateReLU(self):
        # seperate ReLU from proceeding layers
        relus = self.type2names(layer_type='ReLU')
        for relu in relus:
            top = self.net_param.layer[relu][0].top[0]
            bottom = self.net_param.layer[relu][0].bottom[0]
            if top == bottom and top not in self.bns:
                # proceed = self.net_param.layer[relu][0].bottom
                # assert len(proceed) == 1
                self.net_param.ch_top(relu, relu, top)
                for i in self.net_param.layer:
                    if i != relu:
                        self.net_param.ch_bottom(i, relu, bottom)

        new_pt = self.save_pt(prefix = 'relu_separ')
        return new_pt

    def add_bias(self, WPQ):
        for conv in self.type2names('Convolution'):
            if len(self.param(conv)) < 2:
                new_name = underline('bias', conv)
                self.set_conv(conv, new_name=new_name , bias=True)
                no = self.param_shape(conv)[0]
                WPQ[(new_name, 1)] = np.zeros(no)
                if (conv, 0) in WPQ:
                    weights = WPQ.pop((conv, 0))
                else:
                    weights = self.param_data(conv)
                WPQ[(new_name, 0)] = weights
            else:
                pass
        new_pt = self.save_pt(prefix='bias')
        return WPQ, new_pt

    def preprocess(self, merge_bn=True, separate_relu=True, add_bias=True):
        WPQ, pt, model = {}, None, self.caffemodel_dir
        if merge_bn:
            WPQ, pt, model = self.merge_bn()
        if add_bias:
            WPQ, pt = self.add_bias(WPQ)
        if separate_relu:
            pt = self.seperateReLU()
        return WPQ, pt, model

    def decompose(self):
        sd_speed_ratio = self.SD_param.c_ratio
        cd_speed_ratio = self.CD_param.c_ratio
        DP = self.ND_param.DP
        r = self.ND_param.rank
        # now we are not implementing the combination of sd + cd
        if self.SD_param.enable and self.CD_param.enable:
            NotImplementedError

        # make prefix
        prefix = ''
        if self.SD_param.enable:
            rate = str(self.SD_param.c_ratio)
            if '.' in rate:
                rate = rate.replace('.','_')
            prefix += (rate+'_')

        if self.CD_param.enable:
            rate = str(self.CD_param.c_ratio)
            if '.' in rate:
                rate = rate.replace('.','_')
            prefix += (rate+'_')

        if self.ND_param.enable:
            thresh = str(self.ND_param.energy_threshold)
            if '.' in thresh:
                thresh = thresh.replace('.','_')
            prefix += (thresh+'_')

        DEBUG = False
        convs= self.convs
        self.WPQ = dict()

        if self.SD_param.enable:
            rankdic = dict()
            for layer in convs:
                if layer not in self.mask_layers:
                    wshape = self.param_shape(layer)
                    co = wshape[0]
                    ci = wshape[1]
                    h = wshape[2]
                    w = wshape[3]
                    rankdic[layer] = int((ci*co*h*w)/(sd_speed_ratio*(h*ci+w*co)))

        if self.CD_param.enable:
            primedict = dict()
            for layer in convs:
                if layer not in self.mask_layers:
                    wshape = self.param_shape(layer)
                    co = wshape[0]
                    ci = wshape[1]
                    h = wshape[2]
                    w = wshape[3]
                    primedict[layer] = int((ci*co*h*w)/(cd_speed_ratio*(h*w*ci+co)))


        def getX(name):
            _, _, kh, kw = self.param_shape(name)
            x = self.extract_XY(self.bottom_names[name][0], name)
            return np.rollaxis(x.reshape((-1, kh, kw, x.shape[1])), 3, 1).copy()

        def setConv(c, d):
            self.set_param_data(c, d)

        t = Timer()
        decouple_convs = [layer for layer in convs if layer not in self.mask_layers]

        if (self.SD_param.data_driven and self.SD_param.enable) or self.CD_param.enable:
            self.load_frozen()

        for conv in decouple_convs:
            W_shape = self.param_shape(conv)
            """spatial decomposition and network decoupling at the same time"""
            if self.SD_param.enable and self.ND_param.enable:
                # neither of ND and SD could process 1x1 conv, so pass such layers
                if W_shape[2]==1 and W_shape[3]==1:
                    continue

                t.tic()
                print('spatial decomposition for %s' % conv)
                conv_V = underline(conv, 'V')              
                conv_H = underline(conv, 'H')
                rank = rankdic[conv]
                weights = self.param_data(conv)
                dim = weights.shape
                if self.SD_param.data_driven:
                    Y = self._feats_dict[conv] - self.param_b_data(conv)
                    X = getX(conv)
                    V, H, VHr, b = VH_decompose(weights, rank=rank, DEBUG=DEBUG, X=X, Y=Y)
                    self.set_param_b(conv,b)
                else:
                    V, H, VHr = VH_decompose(weights, rank=rank, DEBUG=DEBUG)

                #setConv(conv,VHr)
                if DEBUG:
                    print("W", W_shape)
                    print("V", V.shape)
                    print("H", H.shape)

                t.toc('spatial_decomposition')
                
                t.tic()
                print('decoupling for %s and %s' %(conv_V, conv_H))
                '''decoupling for V'''
                energy_ratio = self.ND_param.energy_threshold
                weights = V # rank, c, h, 1
                
                depth_V, point_V, weights_approx = Network_decouple(weights, energy_threshold=energy_ratio, rank=r, DP=DP)
                decouple_V = False
                V_approx = V
                if len(depth_V) < W_shape[2]:
                    decouple_V = True
                    V_approx = weights_approx
                else:
                    self.WPQ[conv_V] = V # do nothing
                    
                '''decoupling for H'''
                weights = H # rank, c, h, 1
                decouple_H = False
                depth_H, point_H, weights_approx = Network_decouple(weights, energy_threshold=energy_ratio, rank=r, DP=DP)
                H_approx = H
                if len(depth_H) < W_shape[3]:
                    decouple_H = True
                    H_approx = weights_approx
                else:
                    self.WPQ[(conv_H, 0)] = H
                    self.WPQ[(conv_H, 1)] = self.param_b_data(conv)# do nothing
                
                if self.SD_param.data_driven:
                    Vdim = V.shape
                    reH = np.transpose(H_approx, [1,0,2,3]).reshape([rank,-1])
                    reV = V_approx.transpose((1,3,2,0))
                    reV = reV.reshape(Vdim[1]*Vdim[2],-1)
                    VHr = (reV.dot(reH)).reshape([dim[1], dim[2], dim[0], dim[3]])
                    VHr = np.transpose(VHr, [2, 0, 1, 3])
                    setConv(conv,VHr)

                conv_param = caffe_pb2.ConvolutionParameter()
                conv_param.CopyFrom(self.conv_param(conv))                
                # setup V
                if decouple_V:
                    bottom = self.layer_bottom(conv)
                    if len(depth_V) == 1:
                        self.remove(conv)
                        new_p = conv + '_P0'
                        new_d = conv + '_D0'
                        self.WPQ[(new_p,0)] = point_V[0]
                        self.WPQ[(new_d,0)] = depth_V[0]
                        if not DP:
                            self.insert(bottom, new_p, pad=0, kernel_size=1, stride=1, num_output=rank, next_layer=None)
                            self.insert(new_p, new_d, layer_type='ConvolutionDepthwise',next_layer=None)
                            param = self.infer_pad_kernel(depth_V[0],origin_name=None, conv_param=conv_param)
                            self.set_conv(new_d,**param)
                            from_layer = new_d
                        else:
                            self.insert(bottom, new_d, layer_type='ConvolutionDepthwise',next_layer=None)
                            self.insert(new_d, new_p, pad=0, kernel_size=1, stride=1, num_output=rank, next_layer=None)
                            param = self.infer_pad_kernel(depth_V[0],origin_name=None, conv_param=conv_param)
                            self.set_conv(new_d,**param)
                            from_layer = new_p
                    else:
                        sums = conv_V+'_sum'
                        self.insert(conv, sums, layer_type='Eltwise')
                        self.remove(conv)
                        for i in range(len(point_V)):
                            new_p = conv_V + '_P' + str(i)
                            new_d = conv_V + '_D' + str(i)
                            self.WPQ[(new_p,0)] = point_V[i]
                            self.WPQ[(new_d,0)] = depth_V[i]
                            if not DP:
                                self.insert(bottom, new_p, kernel_size=1, stride=1, pad=0, num_output=rank, next_layer=sums)
                                self.insert(new_p, new_d, layer_type='ConvolutionDepthwise',next_layer=sums)
                                param = self.infer_pad_kernel(depth_V[i],origin_name=None, conv_param=conv_param)
                                self.set_conv(new_d,**param)
                            else:
                                self.insert(bottom, new_d, layer_type='ConvolutionDepthwise',next_layer=sums, DP=DP)
                                self.insert(new_d, new_p, kernel_size=1, stride=1, pad=0, num_output=rank, next_layer=sums, DP=DP)
                                param = self.infer_pad_kernel(depth_V[i],origin_name=None, conv_param=conv_param)
                                self.set_conv(new_d,**param)
                        from_layer = sums
                else:
                    V_param = self.infer_pad_kernel(V,origin_name=None, conv_param=conv_param)
                    self.set_conv(conv, new_name=conv_V, **V_param)
                    from_layer = conv_V

                if decouple_H:
                    if len(point_H) == 1:
                        new_p = conv_H + '_P0'
                        new_d = conv_H + '_D0'
                        self.WPQ[(new_p,0)] = point_H[0]
                        self.WPQ[(new_d,0)] = depth_H[0]
                        if conv_param.bias_term:
                            if not DP:
                                self.WPQ[(new_d,1)] = self.param_b_data(conv)
                            else:
                                self.WPQ[(new_p,1)] = self.param_b_data(conv)

                        if not DP:
                            self.insert(from_layer, new_p, kernel_size=1, stride=1, pad=0, num_output=conv_param.num_output, next_layer=None)
                            self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', next_layer=None)
                            H_param = self.infer_pad_kernel(depth_H[0], origin_name=None, conv_param=conv_param)
                            H_param['bias'] = conv_param.bias_term
                            self.set_conv(new_d,**H_param)
                        else:
                            self.insert(from_layer, new_d, layer_type='ConvolutionDepthwise', next_layer=None)
                            self.insert(new_d, new_p, kernel_size=1, stride=1, pad=0, bias=conv_param.bias_term, num_output=conv_param.num_output, next_layer=None)
                            H_param = self.infer_pad_kernel(depth_H[0], origin_name=None, conv_param=conv_param)
                            self.set_conv(new_d,**H_param)
                    else:
                        sums = conv_H+'_sum'
                        self.insert(from_layer, sums, layer_type='Eltwise')
                        for i in range(len(point_H)):
                            new_p = conv_H + '_P' + str(i)
                            new_d = conv_H + '_D' + str(i)
                            self.WPQ[(new_p,0)] = point_H[i]
                            self.WPQ[(new_d,0)] = depth_H[i]
                            if i==0 and conv_param.bias_term:
                                if not DP:
                                    self.WPQ[(new_d,1)] = self.param_b_data(conv)
                                else:
                                    self.WPQ[(new_p,1)] = self.param_b_data(conv)
                            if not DP:
                                self.insert(from_layer, new_p, kernel_size=1, stride=1, pad=0, num_output=conv_param.num_output, next_layer=sums)
                                self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', next_layer=sums)
                                H_param = self.infer_pad_kernel(depth_H[i], origin_name=None, conv_param=conv_param)
                                H_param['bias'] = True if i==0 and conv_param.bias_term else False
                                self.set_conv(new_d,**H_param)
                            else:
                                bias = True if i==0 and conv_param.bias_term else False
                                self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', next_layer=sums, DP=DP)
                                self.insert(from_layer, new_p, kernel_size=1, stride=1, pad=0, bias=bias, num_output=conv_param.num_output, next_layer=sums, DP=DP)
                                H_param = self.infer_pad_kernel(depth_H[i], origin_name=None, conv_param=conv_param)
                                self.set_conv(new_d,**H_param)
                else:
                    self.insert(from_layer, conv_H)
                    H_params = {'bias':True}
                    H_params.update(self.infer_pad_kernel(H, origin_name=None, conv_param=conv_param))
                    self.set_conv(conv_H, **H_params)

                t.toc('decoupling')

            #channel decompostion and network decoupling
            elif self.CD_param.enable and self.ND_param.enable:
                conv_P = underline(conv, 'P')
                conv_new = underline(conv, 'new')
                W_shape = self.param_shape(conv)
                d_prime = primedict[conv]

                t.tic()
                print('channel decomposition for ' + conv)
                branch2cname = '_branch2b'
                branch1name = '_branch1'
                feats_dict, _ = self.extract_features(names=conv, points_dict=self._points_dict, save=1)
                weights = self.param_data(conv)
                Y = feats_dict[conv]
                if not self.resnet or '_branch2b' not in conv:
                    gt_Y = self._feats_dict[conv]
                else:
                    # for res-connection part, use ground truth summation to correct the output
                    error_feat_branch = conv.replace(branch2cname, branch1name)
                    sum_name = conv.replace(branch2cname,'')
                    have_relu = False
                    if error_feat_branch not in self.convs:
                        error_feat_branch = self.sums[self.sums.index(sum_name)-1]
                        have_relu = True

                    error_feats_dict, _ = self.extract_features(names=error_feat_branch, points_dict=self._points_dict, save=1)
                    if have_relu: # relu operation   
                        error_feats_dict[error_feat_branch][error_feats_dict[error_feat_branch]<0]=0

                    gt_Y = self._feats_dict[sum_name] - error_feats_dict[error_feat_branch]

                W1, W2, B, W12, R = ITQ_decompose(Y, gt_Y, weights, d_prime, bias=self.param_b_data(conv), DEBUG=0, Wr=None)

                # set W to low rank W, asymetric solver
                setConv(conv,W12.copy())
                self.set_param_b(conv, B.copy())

                # save W_prime and P params
                W_prime_shape = [d_prime, weights.shape[1], weights.shape[2], weights.shape[3]]
                P_shape = [W2.shape[0], W2.shape[1], 1, 1]
                self.WPQ[(conv_P, 0)] = W2.reshape(P_shape)
                self.WPQ[(conv_P, 1)] = B

                self.insert(conv, conv_P, pad=0, kernel_size=1, bias=True, stride=1)
                weights = W1.reshape(W_prime_shape).copy()
                params = {'bias':True}
                params.update(self.infer_pad_kernel(weights, conv))
                self.set_conv(conv, **params)

                if W_shape[2]*W_shape[3] == 1:
                    #1x1 conv couldn't be decoupled
                    self.WPQ[(conv_new, 0)] = weights
                    self.WPQ[(conv_new, 1)] = np.zeros(d_prime)
                    self.set_conv(conv, new_name=conv_new)
                else:
                    t.toc('channel_decomposition')
                    print('decoupling for %s' % conv)
                    t.tic()
                    energy_ratio = self.ND_param.energy_threshold
                    
                    depth, point, weights_approx = Network_decouple(weights, energy_threshold=energy_ratio, rank=r, DP=DP)
                    dim = weights_approx.shape
                    weights_approx = np.transpose(weights_approx, [1,2,3,0])
                    weights_approx = weights_approx.reshape(-1, dim[0])
                    W12 = weights_approx.dot(R)
            
                    W12 = W12.reshape(dim[1:] + (W12.shape[1],))
                    W12 = np.transpose(W12, [3,0,1,2])
                    setConv(conv, W12.copy())

                    sums = conv+'_sum'
                    #bottom = self.bottom_names[conv][0]
                    bottom = self.layer_bottom(conv)
                    conv_param = caffe_pb2.ConvolutionParameter()
                    conv_param.CopyFrom(self.conv_param(conv))
                    if len(depth) == 1:
                        self.remove(conv)
                        new_p = conv + '_P0'
                        new_d = conv + '_D0'
                        self.WPQ[(new_p,0)] = point[0]
                        self.WPQ[(new_d,0)] = depth[0]
                        if conv_param.bias_term:
                            if not DP:
                                self.WPQ[(new_d,1)] = np.zeros(d_prime)
                            else:
                                self.WPQ[(new_p,1)] = np.zeros(d_prime)
                            bias = True
                        else:
                            bias = False

                        if not DP:
                            self.insert(bottom, new_p, pad=0, kernel_size=1, bias=False, stride=1, \
                                num_output=conv_param.num_output, next_layer=None)
                            self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', \
                                kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                 pad=conv_param.pad[:], bias=bias, num_output=conv_param.num_output, next_layer=None)
                        else:
                            self.insert(bottom, new_d, layer_type='ConvolutionDepthwise', \
                                kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                 pad=conv_param.pad[:], bias=False, num_output=W_shape[1], next_layer=None)
                            self.insert(new_d, new_p, pad=0, kernel_size=1, bias=bias, stride=1, \
                                num_output=conv_param.num_output, next_layer=None)
                            
                    else:
                        self.insert(conv, sums, layer_type='Eltwise') 
                        self.remove(conv)
                        for i in range(len(point)):
                            new_p = conv + '_P' + str(i)
                            new_d = conv + '_D' + str(i)
                            self.WPQ[(new_p,0)] = point[i]
                            self.WPQ[(new_d,0)] = depth[i]
                            if i == 0 and conv_param.bias_term:
                                if not DP:
                                    self.WPQ[(new_d,1)] = np.zeros(d_prime)
                                else:
                                    self.WPQ[(new_p,1)] = np.zeros(d_prime)
                                bias = True
                            else:
                                bias = False

                            if not DP:
                                self.insert(bottom, new_p, pad=0, kernel_size=1, bias=False, stride=1, num_output=conv_param.num_output, next_layer=sums)
                                self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', \
                                    kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                     pad=conv_param.pad[:], bias=bias, num_output=W_shape[1], next_layer=sums)
                            else:
                                self.insert(bottom, new_d, layer_type='ConvolutionDepthwise', \
                                    kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                     pad=conv_param.pad[:], bias=False, num_output=conv_param.num_output, next_layer=sums, DP=DP)
                                self.insert(new_d, new_p, pad=0, kernel_size=1, bias=bias, stride=1, num_output=conv_param.num_output, next_layer=sums, DP=DP)
                                
                    t.toc('decoupling')

            #only channel decomposition
            elif self.CD_param.enable:

                conv_P = underline(conv, 'P')
                conv_new = underline(conv, 'new')
                W_shape = self.param_shape(conv)
                d_prime = primedict[conv]

                t.tic()
                branch2cname = '_branch2b'
                branch1name = '_branch1'
                feats_dict, _ = self.extract_features(names=conv, points_dict=self._points_dict, save=1)
                weights = self.param_data(conv)
                Y = feats_dict[conv]
                if not self.resnet or '_branch2b' not in conv:
                    gt_Y = self._feats_dict[conv]
                else:
                    # for res-connection part, use ground truth summation to correct the output
                    error_feat_branch = conv.replace(branch2cname, branch1name)
                    sum_name = conv.replace(branch2cname,'')
                    have_relu = False
                    if error_feat_branch not in self.convs:
                        error_feat_branch = self.sums[self.sums.index(sum_name)-1]
                        have_relu = True

                    error_feats_dict, _ = self.extract_features(names=error_feat_branch, points_dict=self._points_dict, save=1)
                    if have_relu: # relu operation   
                        error_feats_dict[error_feat_branch][error_feats_dict[error_feat_branch]<0]=0

                    gt_Y = self._feats_dict[sum_name] - error_feats_dict[error_feat_branch]

                Y = feats_dict[conv]
                W1, W2, B, W12, R = ITQ_decompose(Y, gt_Y, weights, d_prime, bias=self.param_b_data(conv), DEBUG=0, Wr=weights)

                # set W to low rank W, asymetric solver
                setConv(conv,W12.copy())
                self.set_param_b(conv, B.copy())

                # save W_prime and P params
                W_prime_shape = [d_prime, weights.shape[1], weights.shape[2], weights.shape[3]]
                P_shape = [W2.shape[0], W2.shape[1], 1, 1]
                self.WPQ[(conv_new, 0)] = W1.reshape(W_prime_shape)
                self.WPQ[(conv_new, 1)] = np.zeros(d_prime)
                self.WPQ[(conv_P, 0)] = W2.reshape(P_shape)
                self.WPQ[(conv_P, 1)] = B

                self.insert(conv, conv_P, pad=0, kernel_size=1, bias=True, stride=1)
                params = self.infer_pad_kernel(self.WPQ[(conv_new,0)], conv)
                params['bias'] = True
                self.set_conv(conv, new_name=conv_new , **params)

                t.toc('channel_decomposition')

            #only spatial decomposition
            elif self.SD_param.enable:
                # neither of ND and SD could process 1x1 conv, so pass such layers
                W_shape = self.param_shape(conv)
                if W_shape[2]==1 and W_shape[3]==1:
                    continue
                    
                conv_V = underline(conv, 'V')              
                conv_H = underline(conv, 'H')
                rank = rankdic[conv]
                
                t.tic()
                weights = self.param_data(conv)
                if self.SD_param.data_driven:
                    if self.conv_param(conv).bias_term:
                        Y = self._feats_dict[conv] - self.param_b_data(conv)
                    else:
                        Y = self._feats_dict[conv]
                    X = getX(conv)
                    V, H, VHr, b = VH_decompose(weights, rank=rank, DEBUG=DEBUG, X=X, Y=Y)
                    self.set_param_b(conv,b)
                else:
                    V, H, VHr = VH_decompose(weights, rank=rank, DEBUG=DEBUG)

                self.WPQ[conv_V] = V
                self.WPQ[(conv_H, 0)] = H
                if self.conv_param(conv).bias_term:
                    self.WPQ[(conv_H, 1)] = self.param_b_data(conv)
                else:
                    self.WPQ[(conv_H, 1)] = np.zeros(weights.shape[0])

                if DEBUG:
                    print("W", W_shape)
                    print("V", V.shape)
                    print("H", H.shape)

                # set W to low rank W, asymetric solver
                setConv(conv,VHr)

                self.insert(conv, conv_H)
                # setup H
                H_params = {'bias':True}
                H_params.update(self.infer_pad_kernel(self.WPQ[(conv_H, 0)], conv))
                self.set_conv(conv_H, **H_params)
                # setup V
                V_params = self.infer_pad_kernel(self.WPQ[conv_V], conv)
                V_params['bias'] = True
                self.set_conv(conv, new_name=conv_V, **V_params)

                t.toc('spatial_decomposition')
                if DEBUG:
                    print("V", H_params)
                    print("H", V_params)

            #only network decoupling
            elif self.ND_param.enable:
                # neither of ND and SD could process 1x1 conv, so pass such layers
                et = self.ND_param.energy_threshold
                weights = self.param_data(conv)
                conv_param = caffe_pb2.ConvolutionParameter()
                conv_param.CopyFrom(self.conv_param(conv))
                t.tic()

                if W_shape[2]==1 and W_shape[3]==1:
                    continue
                    """
                    # only for densenet 1x1 svd test
                    U, V = kernel_svd(weights, ratio=et)
                    rank = U.shape[1]
                    conv_L = underline(conv,'L')
                    conv_R = underline(conv,'R')

                    self.WPQ[conv_R] = V
                    self.WPQ[(conv_L, 0)] = U
                    if conv_param.bias_term:
                        self.WPQ[(conv_L, 1)] = self.param_b_data(conv).copy()

                    self.insert(conv, conv_L, pad=0, kernel_size=1, bias=conv_param.bias_term, stride=1,
                        num_output=conv_param.num_output ,next_layer=None)
                    self.set_conv(conv, new_name=conv_R, bias=False, num_output=rank)
                    """
                else:              
                    depth, point, approx = Network_decouple(weights, energy_threshold=et, rank=r, DP=DP)
                    sums = conv+'_sum'
                    #bottom = self.bottom_names[conv][0]
                    bottom = self.layer_bottom(conv)
                    if len(depth) == 1:
                        self.remove(conv)
                        new_p = conv + '_P0'
                        new_d = conv + '_D0'
                        self.WPQ[(new_p,0)] = point[0]
                        self.WPQ[(new_d,0)] = depth[0]
                        if conv_param.bias_term:
                            if not DP:
                                self.WPQ[(new_d,1)] = self.param_b_data(conv)
                            else:
                                self.WPQ[(new_p,1)] = self.param_b_data(conv)
                            bias = True
                        else:
                            bias = False

                        if not DP:
                            self.insert(bottom, new_p, pad=0, kernel_size=1, bias=False, stride=1, num_output=conv_param.num_output, next_layer=None)
                            self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', \
                                kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                 pad=conv_param.pad[:], bias=bias, num_output=conv_param.num_output, next_layer=None)
                        else:
                            self.insert(bottom, new_d, layer_type='ConvolutionDepthwise', \
                                kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                 pad=conv_param.pad[:], bias=False, num_output=W_shape[1], next_layer=None)
                            self.insert(new_d, new_p, pad=0, kernel_size=1, bias=bias, stride=1, num_output=conv_param.num_output, next_layer=None)
                            
                    else:
                        self.insert(conv, sums, layer_type='Eltwise') 
                        self.remove(conv)
                        for i in range(len(point)):
                            new_p = conv + '_P' + str(i)
                            new_d = conv + '_D' + str(i)
                            self.WPQ[(new_p,0)] = point[i]
                            self.WPQ[(new_d,0)] = depth[i]
                            if i == 0 and conv_param.bias_term:
                                if not DP:
                                    self.WPQ[(new_d,1)] = self.param_b_data(conv)
                                else:
                                    self.WPQ[(new_p,1)] = self.param_b_data(conv)
                                bias = True
                            else:
                                bias = False

                            if not DP:
                                self.insert(bottom, new_p, pad=0, kernel_size=1, bias=False, stride=1, num_output=conv_param.num_output, next_layer=sums)
                                self.insert(new_p, new_d, layer_type='ConvolutionDepthwise', \
                                    kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                     pad=conv_param.pad[:], bias=bias, num_output=conv_param.num_output, next_layer=sums)
                            else:
                                self.insert(bottom, new_d, layer_type='ConvolutionDepthwise', \
                                    kernel_size=conv_param.kernel_size[:], stride=conv_param.stride[:],\
                                     pad=conv_param.pad[:], bias=False, num_output=W_shape[1], next_layer=sums, DP=DP)
                                self.insert(new_d, new_p, pad=0, kernel_size=1, bias=bias, stride=1, num_output=conv_param.num_output, next_layer=sums, DP=DP)                    
                t.toc("network decoupling")
            else:
                pass

        new_pt = self.save_pt(prefix=prefix)
        return self.WPQ, new_pt

    def layer_bottom(self, layer):
        bottom = self.net_param.layer_bottom(layer)
        if isinstance(bottom, list):
            return bottom[0]
        else:
            return bottom
