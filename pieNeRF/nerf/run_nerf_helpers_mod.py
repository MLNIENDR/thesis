"""Helper layers and utilities for the modified NeRF (SIREN variant, sampling, ray/positional helpers)."""

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
searchsorted = torch.searchsorted



# Misc: Hilfsfunktionen für Training & Debugging
img2mse = lambda x, y : torch.mean((x - y) ** 2)                                # MSE zwischen zwei Bildern / Projektionen
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))      # PSNR = Qualitätsmaß für Rekonstruktionen (hoher PSNR --> gut)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)                         # unit8-Konvertierung: Werte [0,1] -> [0,255] (wenn NeRF-Ausgabe als png dargestellt werden soll)
relu = partial(F.relu, inplace=True)                                            # ReLU mit inplace = True spart GPU-Memory

# SIREN-Layer (Sinus-Aktivierung) - wird in pieNeRF NICHT benutzt!
# Siren version not working yet
# TODO: Check right frequencies for our data
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features,
                 is_first=False, is_last=False):
        super().__init__()
        self.omega_0 = 30
        self.is_first = is_first 
        self.is_last = is_last
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
    
    def init_weights(self):
        with torch.no_grad():
            num_input = self.linear.weight.size(-1)
            if self.is_first:
                self.linear.weight.uniform_(-1 / num_input, 1 / num_input)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / num_input) / self.omega_0, np.sqrt(6 / num_input) / self.omega_0)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.omega_0 * x)


# NeRF mit SIREN-Activations - wird NICHT verwendet
class NeRF_Siren(nn.Module):    
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF_Siren, self).__init__()
        self.D = D
        self.W = W
        self.w_0 = 30
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, is_first=True)] + [SineLayer(W, W) if i not in self.skips else SineLayer(W + input_ch, W) for i in range(D-1)])

        self.views_linears = nn.ModuleList([SineLayer(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = SineLayer(W, W, is_last=True)
            self.alpha_linear = SineLayer(W, 1, is_last=True)
            self.rgb_linear = SineLayer(W//2, 3, is_last=True)
        else:
            self.output_linear = SineLayer(W, output_ch, is_last=True)
    
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# Positional Encoding (Fourier Features) - wichtig für NeRF
class Embedder:
    """Baut mehrere Frequenz-Basisfunktionen sin(kx), cos(kx)
    für positionsbezogene Encodings (wie im NeRF Paper)"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:                        # Originalwerte behalten (x selbst)
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']                 # Frequenzbänder bestimmen (0, ..., 2^max_freq)
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:                                 # sin(freq*x), cos(freq*x)
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        """Alle Embeddings konkatenieren"""
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    """Hilfsfunktion: Erzeugt ein Embedder-Objekt + Lambda-Funktion für den Encoder"""
    if i == -1:
        return nn.Identity(), 3                                 # kein positional encoding
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# NeRF-MLP (hier werden Punkte -> Werte gemappt)
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ Standard-NeRF-MLP mit optionaler Viewdir-Konditionierung"""
        super(NeRF, self).__init__()
        self.D = D                                  # Tiefe: Anz. Punkt-Layer
        self.W = W                                  # Breite: Anz. Neuronen pro Layer
        self.input_ch = input_ch                    # Dim der Positions-Eingabe (nach PosEnc)
        self.input_ch_views = input_ch_views        # Dim der Viewdir-Eingabe (nach PosEnc)    
        self.skips = skips                          # Layer-Indizes mit Skip-Connection    
        self.use_viewdirs = use_viewdirs            # falls True: RGB abh. von Blickrichtung
        
        # MLP für 3D-Punkte (mit opt. Skip-Connections)
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        # View-MLP (offizielle NeRF Implementierung: ein Layer)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # Falls Blickrichtungen genutzt: Feature/Alpha/RGB getrennt
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:               # sonst direkt Gesamt-Output (z.B. [sigma, rgb] oder Emission, ...)
            self.output_linear = nn.Linear(W, output_ch)
        
        # --- Emission-NeRF Init Fix: leichte positive Startwerte ---
        with torch.no_grad():
            self.output_linear.bias.fill_(1e-3)    # kleiner positiver Bias
            self.output_linear.weight.mul_(0.01)   # kleine Gewichtsamplitude (nahe 0)

    def forward(self, x):
        """x: [..., input_ch + input_ch_views]
           = (positional-encodete Positionen || positional-encodete Viewdirs)"""
        # Aufteilung: Punkt- und Viewdir-Anteil
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        relu = partial(F.relu, inplace=True)

        # Punkt-MLP durchlaufen
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)
            if i in self.skips:
                # Skip-Connection: ursprüngliche Eingabe wieder anhängen
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # Dichte / Alpha direkt aus Punkt-Features
            alpha = self.alpha_linear(h)
            # Feature-Vektor für weitere Verarbeitung
            feature = self.feature_linear(h)
            # Feature + Viewdir als Input für RGB-MLP
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = relu(h)

            rgb = self.rgb_linear(h)                            # Farbausgabe
            outputs = torch.cat([rgb, alpha], -1)               # Output: [rgb, alpha]
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        """NICHT GENUTZT: Lädt Gewichte aus Keras-NeRF-Modell (Kompatibilitäts-Helfer)"""
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Punkt-MLP-Gewichte laden
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Feature-Layer
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # View-MLP
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # RGB-MLP
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Alpha-MLP
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers - Funktionen, um aus Kamera-Parametern Rays zu bauen
def get_rays(H, W, focal, c2w):
    """NICHT GENUTZT: Erzeugt für eine pinhole-Kamera alle Rays (Ursprung + Richtung) eines HxW-Bildes"""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))         # Pixelgitter ni Bildkoordinaten
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)      # Richtungen im Kameraraum (z zeigt nach -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)                       # in Welt-Raum rotieren
    rays_o = c2w[:3,-1].expand(rays_d.shape)                                            # Kameraursprung ins Weltkoordinatensystem (identisch für alle Rays)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    """NumPy-Version von get_rays (z.B. für Precomputing / CPU)"""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """NICHT GENUTZT: Transformiert Rays in den Normalized Device Coordinate (NDC)-Raum,
    wie im Original-NeRF für bessere numerische Stabilität"""
    # Ursprung auf near-plane verschieben
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projektion in NDC-Koordinatensystem
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


def get_rays_ortho(H, W, c2w, size_h, size_w):
    """Erzeugt orthografische Rays: parallele Richtungen, Ursprünge liegen auf einer Ebene 
    mit physikalischer Größe size_w x size_h in Weltkoordinaten.
    Wird für SPECT-AP/PA-Projektionen genutzt"""
    device = c2w.device
    dtype = c2w.dtype

    # Parallele Strahlen entlang der -Z-Achse des Kameraraums
    rays_d = -c2w[:3, 2].view(1, 1, 3).expand(H, W, -1)

    # Erzeuge ein kartesisches Gitter in der Bildebene (Weltmaße = size_w/size_h)
    xs = torch.linspace(-0.5 * size_w, 0.5 * size_w, W, device=device, dtype=dtype)
    ys = torch.linspace(-0.5 * size_h, 0.5 * size_h, H, device=device, dtype=dtype)
    try:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:  # older torch without indexing kwarg
        grid_y, grid_x = torch.meshgrid(ys, xs)
    zeros = torch.zeros_like(grid_x)
    rays_o_cam = torch.stack([grid_x, grid_y, zeros], dim=-1)

    # Drehe in den Welt-Raum und addiere Kameraposition
    rays_o = torch.sum(rays_o_cam[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = rays_o + c2w[:3, -1].view(1, 1, 3)

    return rays_o, rays_d


# Hierarchisches Sampling (importance sampling entlang der Rays)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    """Zieht N_samples neue Tiefen-Samples entlang eines Rays gemäß einer diskreten PDF,
    um feiner in wichtigen Bereichen (hohe weights) zu sampeln"""
    weights = weights + 1e-5                                    # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)        # Normieren: weights -> pdf (Probability Density Function)
    cdf = torch.cumsum(pdf, -1)                                 # CDF berechnen (cumulative distribution function)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    # Gleichverteilte Zufallszahlen in [0,1] ziehen
    if det:
        u = torch.linspace(0., 1., steps=N_samples)             # deterministisches Sampling (fixe Positionen)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])      # zufälliges Sampling

    # Optional: für Tests reproduzierbar mit NumPy
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Inverse CDF: für jedes u das passende Intervall in CDF finden
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # CDF- und Bin-Werte der gewählten Intervalle einsammeln
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # Lineare Interpolation innerhalb des Intervalls
    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples
