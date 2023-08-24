"""Policies: abstract base class and concrete implementations."""

from typing import TypeVar

from flax.struct import TNode

T = TypeVar('T')

from typing import Any, Callable

from flax import core
from flax import struct
import optax


# @flax.struct.dataclass
# class Model:
# 	step: int
# 	apply_fn: Callable[..., Any] = flax.struct.field(pytree_node=False)
# 	params: Params
# 	batch_stats: Union[Params]
# 	tx: Optional[optax.GradientTransformation] = flax.struct.field(pytree_node=False)
# 	opt_state: Optional[optax.OptState] = None
# 	# model_cls: Type = None
#
# 	@classmethod
# 	def create(
# 		cls,
# 		model_def: nn_module.Module,
# 		inputs: Sequence[jnp.ndarray],
# 		tx: Optional[optax.GradientTransformation] = None,
# 		**kwargs
# 	) -> 'Model':
#
# 		variables = model_def.init(*inputs)
#
# 		_, params = variables.pop('params')
#
# 		"""
# 		NOTE:
# 			Here we unfreeze the parameter.
# 			This is because some optimizer classes in optax must receive a dict, not a frozendict, which is annoying.
# 			https://github.com/deepmind/optax/issues/160
# 			And ... if we can access to the params, then why it should be freezed ?
# 		"""
# 		# NOTE : Unfreeze the parameters !!!!!
# 		params = params.unfreeze()
#
# 		# Frozendict's 'pop' method does not support default value. So we use get method instead.
# 		batch_stats = variables.get("batch_stats", None)
#
# 		if tx is not None:
# 			opt_state = tx.init(params)
# 		else:
# 			opt_state = None
#
# 		return cls(
# 			step=1,
# 			apply_fn=model_def.apply,
# 			params=params,
# 			batch_stats=batch_stats,
# 			tx=tx,
# 			opt_state=opt_state,
# 			**kwargs
# 		)
#
# 	def __call__(self, *args, **kwargs):
# 		return self.apply_fn({"params": self.params}, *args, **kwargs)
#
# 	def apply_gradient(
# 		self,
# 		loss_fn: Optional[Callable[[Params], Any]] = None,
# 		grads: Optional[Any] = None,
# 		has_aux: bool = True
# 	) -> Union[Tuple['Model', Any], 'Model']:
#
# 		assert ((loss_fn is not None) or (grads is not None), 'Either a loss function or grads must be specified.')
#
# 		if grads is None:
# 			grad_fn = jax.grad(loss_fn, has_aux=has_aux)
# 			if has_aux:
# 				grads, aux = grad_fn(self.params)
# 			else:
# 				grads = grad_fn(self.params)
# 		else:
# 			assert (has_aux, 'When grads are provided, expects no aux outputs.')
#
# 		updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
# 		new_params = optax.apply_updates(self.params, updates)
# 		new_model = self.replace(step=self.step + 1, params=new_params, opt_state=new_opt_state)
#
# 		if has_aux:
# 			return new_model, aux
# 		else:
# 			return new_model
#
# 	def save_dict_from_path(self, save_path: str) -> Params:
# 		os.makedirs(os.path.dirname(save_path), exist_ok=True)
# 		with open(save_path, 'wb') as f:
# 			f.write(flax.serialization.to_bytes(self.params))
# 		return self.params
#
# 	def load_dict_from_path(self, load_path: str) -> "Model":
# 		with open(load_path, 'rb') as f:
# 			params = flax.serialization.from_bytes(self.params, f.read())
# 		return self.replace(params=params)
#
# 	def save_batch_stats_from_path(self, save_path: str) -> Params:
# 		os.makedirs(os.path.dirname(save_path), exist_ok=True)
# 		with open(save_path, 'wb') as f:
# 			f.write(flax.serialization.to_bytes(self.batch_stats))
# 		return self.batch_stats
#
# 	def load_batch_stats_from_path(self, load_path: str) -> "Model":
# 		with open(load_path, 'rb') as f:
# 			batch_stats = flax.serialization.from_bytes(self.batch_stats, f.read())
# 		return self.replace(batch_stats=batch_stats)
#
# 	def load_dict(self, params: bytes) -> 'Model':
# 		params = flax.serialization.from_bytes(self.params, params)
# 		return self.replace(params=params)

class Model(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

    Synopsis::

        state = TrainState.create(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx)
        grad_fn = jax.grad(make_loss_fn(state.apply_fn))
        for batch in data:
          grads = grad_fn(state.params, batch)
          state = state.apply_gradients(grads=grads)

    Note that you can easily extend this dataclass by subclassing it for storing
    additional data (e.g. additional variable collections).

    For more exotic usecases (e.g. multiple optimizers) it's probably best to
    fork the class and modify it.

    Args:
      step: Counter starts at 0 and is incremented by every call to
        `.apply_gradients()`.
      apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
        convenience to have a shorter params list for the `train_step()` function
        in your training loop.
      params: The parameters to be updated by `tx` and used by `apply_fn`.
      tx: An Optax gradient transformation.
      opt_state: The state for `tx`.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
