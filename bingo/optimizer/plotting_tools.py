import numpy as np
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
font_size = 18
mpl.rcParams["font.size"] = font_size
mpl.rcParams["xtick.labelsize"] = font_size
mpl.rcParams["ytick.labelsize"] = font_size
mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
import qutip as qt

def wigner(rho, xvec, yvec=None, g=2):
    if yvec is None:
        yvec = xvec
    # return (np.pi / 2.0) * qt.wigner(rho, xvec, yvec, g=2)
    N = rho.dims[0][0]
    max_radius = np.sqrt(np.max(xvec * g / 2.0) ** 2 + np.max(yvec * g / 2.0) ** 2)
    if N < 0.8 * max_radius**2:
        print(
            "Warning: calculating Wigner with N = %d and max radius=%.3f"
            % (N, max_radius)
        )
    return qt.wigner(rho, xvec, yvec, g=g)

#%%
# TODO: update the tensorflow side of things.

#%% Complex CF to Wigner. For now, assuming square with same # of sample points in x and y, symmetric around zero
def CF2W(CF, betas_I, zero_pad=True, padding=20, betas_Q=None):
    betas_Q = betas_I if betas_Q is None else betas_Q
    if zero_pad:
        dbeta_I = betas_I[1] - betas_I[0]
        new_min_I = betas_I[0] - padding * dbeta_I
        new_max_I = betas_I[0] + padding * dbeta_I
        betas_I = np.pad(
            betas_I,
            (padding, padding),
            mode="linear_ramp",
            end_values=(new_min_I, new_max_I),
        )
        dbeta_Q = betas_Q[1] - betas_Q[0]
        new_min_Q = betas_Q[0] - padding * dbeta_Q
        new_max_Q = betas_Q[0] + padding * dbeta_Q
        betas_Q = np.pad(
            betas_Q,
            (padding, padding),
            mode="linear_ramp",
            end_values=(new_min_Q, new_max_Q),
        )
        CF = np.pad(CF, (padding, padding), mode="constant")
    N_I = len(betas_I)
    dbeta_I = betas_I[1] - betas_I[0]
    N_Q = len(betas_Q)
    dbeta_Q = betas_Q[1] - betas_Q[0]
    W = (dbeta_I * dbeta_Q / np.pi**2) * (np.fft.fft2(a=CF)).T
    # recenter
    W = np.fft.fftshift(W)
    # todo...check this N_I vs N_Q
    for j in range(N_I):
        for k in range(N_Q):
            W[j, k] = (
                np.exp(1j * np.pi * (j + k)) * W[j, k]
            )  # todo: single matrix multiply
    alpha0_Q = np.pi / (2 * dbeta_I)

    alpha0_I = np.pi / (2 * dbeta_Q)
    alphas_Q = np.linspace(-alpha0_Q, alpha0_Q, N_Q)
    alphas_I = np.linspace(
        -alpha0_I, alpha0_I, N_I
    )  # is it true that it ends at alpha0? or alpha0-dalpha?
    return W, alphas_I, alphas_Q


# todo: can make this faster...
# Use symmetry
# and use diagonalizaed construction of displacement ops
def characteristic_function(rho, xvec, yvec=None):
    yvec = xvec if yvec is None else yvec
    N = rho.dims[0][0]
    a = qt.destroy(N)

    X, Y = np.meshgrid(xvec, yvec)

    def CF(beta):
        return qt.expect((beta * a.dag() - np.conj(beta) * a).expm(), rho)

    CF = np.vectorize(CF)
    Z = CF(X + 1j * Y)
    return Z


# note: for now, only working with pure states
# todo: can easily extend to rho.
def characteristic_function_tf(psi, betas):
    from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer

    N = psi.dims[0][0]
    # dummy opt object for calculation
    params = {"optimization_type": "calculation", "N_cav": N}
    opt = BatchOptimizer(**params)
    return opt.characteristic_function(psi=psi, betas=betas)


def characteristic_function_rho_tf(rho, betas):
    from ECD_control.ECD_optimization.batch_optimizer import BatchOptimizer

    N = rho.dims[0][0]
    # dummy opt object for calculation
    params = {"optimization_type": "calculation", "N_cav": N}
    opt = BatchOptimizer(**params)
    return opt.characteristic_function_rho(rho=rho, betas=betas)


def plot_expect(states):
    expects = expect(states)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6))
    plot1 = ["sx", "sy", "sz"]
    plot2 = ["n", "a"]
    for n, p in enumerate([plot1, plot2]):
        for name in p:
            e = expects[name]
            if name == "a":
                axs[n].plot(np.real(e), "-", label="re(a)")
                axs[n].plot(np.imag(e), "-", label="re(a)")
            else:
                axs[n].plot(e, "-", label=name)
        axs[n].grid()
        axs[n].legend(frameon=False)


def plot_expect_displace(states, alphas):
    expects = expect_displaced(states, alphas)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6))
    plot1 = ["sx", "sy", "sz"]
    plot2 = ["n", "a"]
    for n, p in enumerate([plot1, plot2]):
        for name in p:
            e = expects[name]
            if name == "a":
                axs[n].plot(np.real(e), "-", label="re(a)")
                axs[n].plot(np.imag(e), "-", label="im(a)")
            else:
                axs[n].plot(e, "-", label=name)
        axs[n].grid()
        axs[n].legend(frameon=False)


def plot_cf(
    xvec_data,
    data_cf,
    yvec_data=None,
    sample_betas=None,
    v=1.0,
    title="",
    grid=True,
    bwr=False,
    axs=None,
    labels=True,
    figsize=(8, 6),
):
    dx_data = xvec_data[1] - xvec_data[0]
    yvec_data = xvec_data if yvec_data is None else yvec_data
    dy_data = yvec_data[1] - yvec_data[0]
    extent_data = (
        xvec_data[0] - dx_data / 2.0,
        xvec_data[-1] + dx_data / 2.0,
        yvec_data[0] - dy_data / 2.0,
        yvec_data[-1] + dy_data / 2.0,
    )
    if axs is None:
        fig, axs = plt.subplots(
            nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize
        )
    cmap = "bwr" if bwr else "seismic"
    axs[0].imshow(
        np.real(data_cf),
        origin="lower",
        extent=extent_data,
        cmap=cmap,
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    if sample_betas is not None:
        axs[0].scatter(
            np.real(sample_betas), np.imag(sample_betas), marker="x", color="black"
        )
    if grid:
        axs[0].grid()
    if labels:
        axs[0].set_xlabel("Re(beta)")
        axs[0].set_ylabel("Im(beta)")
        axs[0].set_title("Real")
        axs[0].set_axisbelow(True)
    axs[1].imshow(
        np.imag(data_cf),
        origin="lower",
        extent=extent_data,
        cmap=cmap,
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    if grid:
        axs[1].grid()
    if labels:
        axs[1].set_xlabel("Re(beta)")
        axs[1].set_ylabel("Im(beta)")
        axs[1].set_title("Imag")
        axs[1].set_axisbelow(True)

        fig.suptitle(title)
        fig.tight_layout()


# for now, only for real part.
def plot_cf_sampled(
    sample_betas,
    C_vals,
    beta_extent_real=[-5, 5],
    beta_extent_imag=[-5, 5],
    v=1.0,
):
    dummy_xvec = np.linspace(beta_extent_real[0], beta_extent_real[1], 11)
    dummy_yvec = np.linspace(beta_extent_real[0], beta_extent_real[1], 11)
    dummy_data = np.zeros(shape=(len(dummy_xvec), len(dummy_yvec)))

    dx = dummy_xvec[1] - dummy_xvec[0]
    dy = dummy_yvec[1] - dummy_yvec[0]
    extent = (
        dummy_xvec[0] - dx / 2.0,
        dummy_xvec[-1] + dx / 2.0,
        dummy_xvec[0] - dy / 2.0,
        dummy_xvec[-1] + dy / 2.0,
    )
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(5, 5))
    ax.imshow(
        np.real(dummy_data),
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    ax.scatter(
        np.real(sample_betas),
        np.imag(sample_betas),
        marker="o",
        s=20,
        c=C_vals,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
    )


def plot_cf_and_marginals(xvec_data, data_cf, yvec_data=None, v=1.0, title=""):
    dx_data = xvec_data[1] - xvec_data[0]
    yvec_data = xvec_data if yvec_data is None else yvec_data
    dy_data = yvec_data[1] - yvec_data[0]
    extent_data = (
        xvec_data[0] - dx_data / 2.0,
        xvec_data[-1] + dx_data / 2.0,
        yvec_data[0] - dy_data / 2.0,
        yvec_data[-1] + dy_data / 2.0,
    )
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=False, figsize=(10, 14)
    )
    axs[0, 0].imshow(
        np.real(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 0].grid()
    axs[0, 0].set_xlabel("Re(beta)")
    axs[0, 0].set_ylabel("Im(beta)")
    axs[0, 0].set_title("Real")
    axs[0, 1].imshow(
        np.imag(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 1].grid()
    axs[0, 1].set_xlabel("Re(beta)")
    axs[0, 1].set_ylabel("Im(beta)")
    axs[0, 1].set_title("Imag")

    mid_idx_data_I = int(len(xvec_data) / 2)
    mid_idx_data_Q = int(len(yvec_data) / 2)

    axs[1, 0].plot(
        xvec_data, np.real(data_cf[mid_idx_data_Q, :]), ".", color="blue", label="re"
    )
    axs[1, 0].plot(
        xvec_data, np.imag(data_cf[mid_idx_data_Q, :]), ".", color="red", label="im"
    )
    axs[1, 0].grid()
    axs[1, 0].legend(frameon=False)
    axs[1, 0].set_xlabel("Re(beta)")
    axs[1, 0].set_ylabel("CF")
    axs[1, 0].set_title("q cut")
    axs[1, 1].plot(
        yvec_data, np.real(data_cf[:, mid_idx_data_I]), ".", color="blue", label="re"
    )
    axs[1, 1].plot(
        yvec_data, np.imag(data_cf[:, mid_idx_data_I]), ".", color="red", label="im"
    )
    axs[1, 1].grid()
    axs[1, 1].legend(frameon=False)
    axs[1, 1].set_xlabel("Im(beta)")
    axs[1, 1].set_ylabel("CF")
    axs[1, 1].set_title("p cut")
    fig.suptitle(title)
    # fig.tight_layout()


def plot_data_and_model_cf(
    xvec_data,
    data_cf,
    xvec_model,
    model_cf,
    yvec_data=None,
    yvec_model=None,
    residuals=False,
    residual_multiplier=10,
    v=1.0,
    title="",
):
    dx_data = xvec_data[1] - xvec_data[0]
    yvec_data = xvec_data if yvec_data is None else yvec_data
    dy_data = yvec_data[1] - yvec_data[0]
    extent_data = (
        xvec_data[0] - dx_data / 2.0,
        xvec_data[-1] + dx_data / 2.0,
        yvec_data[0] - dy_data / 2.0,
        yvec_data[-1] + dy_data / 2.0,
    )
    dx_model = xvec_model[1] - xvec_model[0]
    yvec_model = xvec_model if yvec_model is None else yvec_model
    dy_model = yvec_model[1] - yvec_model[0]
    extent_model = (
        xvec_model[0] - dx_model / 2.0,
        xvec_model[-1] + dx_model / 2.0,
        yvec_model[0] - dy_model / 2.0,
        yvec_model[-1] + dy_model / 2.0,
    )
    mid_idx_data_I = int(len(xvec_data) / 2)
    mid_idx_model_I = int(len(yvec_model) / 2)
    mid_idx_data_Q = int(len(xvec_data) / 2)
    mid_idx_model_Q = int(len(yvec_model) / 2)
    nrows = 3 if residuals else 2
    ysize = 14 if residuals else 10
    fig, axs = plt.subplots(
        nrows=nrows, ncols=3, sharex=False, sharey=False, figsize=(16, ysize)
    )
    axs[0, 0].imshow(
        np.real(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 0].grid()
    axs[0, 0].set_xlabel("Re(beta)")
    axs[0, 0].set_ylabel("Im(beta)")
    axs[0, 0].set_title("Real data")
    axs[0, 1].imshow(
        np.imag(data_cf),
        origin="lower",
        extent=extent_data,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[0, 1].grid()
    axs[0, 1].set_xlabel("Re(beta)")
    axs[0, 1].set_ylabel("Im(beta)")
    axs[0, 1].set_title("Imag data")
    axs[1, 0].imshow(
        np.real(model_cf),
        origin="lower",
        extent=extent_model,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[1, 0].grid()
    axs[1, 0].set_xlabel("Re(beta)")
    axs[1, 0].set_ylabel("Im(beta)")
    axs[1, 0].set_title("Real model")
    axs[1, 1].imshow(
        np.imag(model_cf),
        origin="lower",
        extent=extent_model,
        cmap="seismic",
        vmin=-v,
        vmax=+v,
        interpolation=None,
    )
    axs[1, 1].grid()
    axs[1, 1].set_xlabel("Re(beta)")
    axs[1, 1].set_ylabel("Im(beta)")
    axs[1, 1].set_title("Imag model")
    axs[0, 2].plot(
        xvec_data, np.real(data_cf[mid_idx_data_Q, :]), ".", color="blue", label="re"
    )
    axs[0, 2].plot(xvec_model, np.real(model_cf[mid_idx_model_Q, :]), color="blue")
    axs[0, 2].plot(
        xvec_data, np.imag(data_cf[mid_idx_data_Q, :]), ".", color="red", label="im"
    )
    axs[0, 2].plot(xvec_model, np.imag(model_cf[mid_idx_model_Q, :]), color="red")
    axs[0, 2].grid()
    axs[0, 2].legend(frameon=False)
    axs[0, 2].set_xlabel("Re(beta)")
    axs[0, 2].set_ylabel("CF")
    axs[0, 2].set_title("q cut")
    axs[1, 2].plot(
        yvec_data, np.real(data_cf[:, mid_idx_data_I]), ".", color="blue", label="re"
    )
    axs[1, 2].plot(yvec_model, np.real(model_cf[:, mid_idx_model_I]), color="blue")
    axs[1, 2].plot(
        yvec_data, np.imag(data_cf[:, mid_idx_data_I]), ".", color="red", label="im"
    )
    axs[1, 2].plot(yvec_model, np.imag(model_cf[:, mid_idx_model_I]), color="red")
    axs[1, 2].grid()
    axs[1, 2].legend(frameon=False)
    axs[1, 2].set_xlabel("Im(beta)")
    axs[1, 2].set_ylabel("CF")
    axs[1, 2].set_title("p cut")
    # to plot residuals, the data and model must be sampled the same
    if residuals:
        real_residual = np.real(data_cf) - np.real(model_cf)
        imag_residual = np.imag(data_cf) - np.imag(model_cf)
        axs[2, 0].imshow(
            residual_multiplier * real_residual,
            origin="lower",
            extent=extent_data,
            cmap="seismic",
            vmin=-v,
            vmax=+v,
            interpolation=None,
        )
        axs[2, 0].grid()
        axs[2, 0].set_xlabel("Re(beta)")
        axs[2, 0].set_ylabel("Im(beta)")
        axs[2, 0].set_title("Real residual %dx" % residual_multiplier)
        axs[2, 1].imshow(
            residual_multiplier * imag_residual,
            origin="lower",
            extent=extent_data,
            cmap="seismic",
            vmin=-v,
            vmax=+v,
            interpolation=None,
        )
        axs[2, 1].grid()
        axs[2, 1].set_xlabel("Re(beta)")
        axs[2, 1].set_ylabel("Im(beta)")
        axs[2, 1].set_title("Imag residual %dx" % residual_multiplier)
    fig.suptitle(title)
    fig.tight_layout()


def plot_wigner(psi, xvec=np.linspace(-5, 5, 151), ax=None, grid=True, invert=False):
    W = wigner(psi, xvec)
    s = -1 if invert else +1
    plot_wigner_data(s * W, xvec, ax=ax, grid=grid)


def plot_wigner_combined(
    psi, N_cav=50, xvec=np.linspace(-5, 5, 41), ax=None, grid=True, invert=False
):
    g = qt.tensor(qt.basis(2, 0), qt.identity(N_cav))
    e = qt.tensor(qt.basis(2, 1), qt.identity(N_cav))
    psi = qt.ket2dm(psi)
    psigg = g.dag() * psi * g
    psige = g.dag() * psi * e
    psieg = e.dag() * psi * g
    psiee = e.dag() * psi * e
    Wgg = wigner(psigg, xvec)
    Wge = wigner(psige, xvec)
    Weg = wigner(psieg, xvec)
    Wee = wigner(psiee, xvec)
    s = -1 if invert else +1
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)

    plot_wigner_data(s * Wgg, xvec, ax=axs[0, 0], grid=grid)
    plot_wigner_data(s * Wge, xvec, ax=axs[0, 1], grid=grid)
    plot_wigner_data(s * Weg, xvec, ax=axs[1, 0], grid=grid)
    plot_wigner_data(s * Wee, xvec, ax=axs[1, 1], grid=grid)
    for ax1 in axs:
        for ax in ax1:
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])


def plot_wigner_data(
    W,
    xvec=np.linspace(-5, 5, 101),
    ax=None,
    grid=True,
    yvec=None,
    cut=0,
    vmin=-2 / np.pi,
    vmax=+2 / np.pi,
):
    yvec = xvec if yvec is None else yvec
    if cut > 0:
        xvec = xvec[cut:-cut]
        yvec = yvec[cut:-cut]
        W = W[cut:-cut, cut:-cut]
    dx = xvec[1] - xvec[0]
    dy = yvec[1] - yvec[0]
    extent = (
        xvec[0] - dx / 2.0,
        xvec[-1] + dx / 2.0,
        yvec[0] - dy / 2.0,
        yvec[-1] + dy / 2.0,
    )
    if ax is None:
        plt.figure()
        ax = plt.gca()
    ax.imshow(
        W,
        origin="lower",
        extent=extent,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        interpolation=None,
    )
    # plt.colorbar()
    if grid:
        ax.grid()


def plot_wigner_data_marginals(w_data, xvec, yvec=None, grid=True, norms=True, cut=0):
    yvec = xvec if yvec is None else yvec
    if cut > 0:
        xvec = xvec[cut:-cut]
        yvec = yvec[cut:-cut]
        w_data = w_data[cut:-cut, cut:-cut]
    w_marginal_q = wigner_marginal(w_data, xvec, yvec)
    w_marginal_p = wigner_marginal(w_data, xvec, yvec, I=False)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(xvec, w_marginal_q, ".")
    axs[1].plot(yvec, w_marginal_p, ".")
    if grid:
        axs[0].grid()
        axs[1].grid()
    q_title = "P(q)"
    p_title = "P(p)"
    if norms:
        dx = xvec[1] - xvec[0]
        dy = yvec[1] - yvec[0]
        norm_q = np.sum(w_marginal_q) * dx
        norm_p = np.sum(w_marginal_p) * dy
        q_title += " norm: %.3f" % norm_q
        p_title += " norm: %.3f" % norm_p
    axs[0].set_title(q_title)
    axs[1].set_title(p_title)


def plot_probability_distribution(
    rho,
    xvec=np.linspace(-6, 6, 201),
    yvec=None,
    axis=0,
    wigner_units=False,
    normalize=True,
    ax=None,
):
    from Bosonic_tools.analysis_tools import probability_distribution

    P = probability_distribution(rho, xvec, yvec, axis, wigner_units, normalize)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(xvec, P)
