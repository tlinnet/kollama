from contextlib import contextmanager
import knime.extension as knext


@contextmanager
def override_tqdm_with_ctx(ctx: knext.ExecutionContext = None):
    from tqdm.auto import tqdm

    def ctx_display(self, msg=None, pos=None):
        if ctx.is_canceled():
            raise RuntimeError("Node execution was canceled.")
        ctx.set_progress(self.n / self.total)
        return True

    real_display = tqdm.display

    try:
        tqdm.display = ctx_display
        yield
    finally:
        # Restore the original display
        tqdm.display = real_display
