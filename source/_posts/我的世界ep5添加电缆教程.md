---
title: 我的世界Forge ep5：添加线缆
date: 2023-10-27 18:10:25
tags: 
- 我的世界
- 模组
cover: https://w.wallhaven.cc/full/yx/wallhaven-yx9dvd.jpg
---
# ep5

## 声明

本文章翻译自mcjty的教程，[源地址](https://www.mcjty.eu/docs/1.20/ep5#introduction),侵删

## link

- [视频](https://youtu.be/WUhet8dOlAs)
- [Github地址](https://github.com/McJty/Tut4_3Power)

## 介绍

本章是ep4的继续，在ep4中，增加了能发电的发电机和一个消耗电量的方块。如果你把他们彼此放在一起，你就可以让发电机给电池充电，然而，我们还没有办法远距离传输电量，在这节中，我们增加一种简单的电缆系统，它可以将电力从任何的发电机或者电池传输到任意需要电力的机器上。这不是一个完整的系统，这是一个简单的系统达到我们的目的。同时我们希望这个线缆是一个含水方块

- 烘焙模型
- 自定义模型加载器
- 复杂的形状
- 含水方块

## 烘焙模型

可以使用一个简单的json制作一个线缆系统，然而，这样会导致生成很多的json组合，同时我们希望可以模拟其他的方块，这是Json模型无法做到的，所以我们使用烘焙模型，烘焙模型是一个我们可以使用代码生成模型的系统，在这一系统，这样做会有更多的工作量，但是也更加灵活

### 线缆

![cables-1e682971f988e084ee146460c1910911](https://picst.sunbangyan.cn/2023/10/27/805bccded2b6f545898a4a05a2b24f44.png)

#### 连接类型枚举

这是一个枚举类，用于表示某个方向上的链接类型，有以下三种类型的数值:`CABLE`,`BLOCK`,`NONE`

- `CABLE`:表示这个方向有一个电缆
- `BLOCK`:表示这个方向有一个有一个方块
- `NONE`:表示这个方向什么也没有

```java

public enum ConnectorType implements StringRepresentable {
    NONE,
    CABLE,
    BLOCK;

    public static final ConnectorType[] VALUES = values();

    @Override
    @Nonnull
    public String getSerializedName() {
        return name().toLowerCase();
    }
}


```

#### 线缆方块

电缆也是一个方块，所以我们需要添加一个新的方块，我们叫他CableBlock，这个类中有比较多的带， 所以这里我们分为几个部分讲。

首先需要6个枚举用于指明某个方向上是由含有线缆或者方块，`FACEDID`是一个特殊的模型属性，我们使用这个属性指出我们正在模仿另一个块的面

```java

public class CableBlock extends Block implements SimpleWaterloggedBlock, EntityBlock {

    // Properties that indicate if there is the same block in a certain direction.
    public static final EnumProperty<ConnectorType> NORTH = EnumProperty.<ConnectorType>create("north", ConnectorType.class);
    public static final EnumProperty<ConnectorType> SOUTH = EnumProperty.<ConnectorType>create("south", ConnectorType.class);
    public static final EnumProperty<ConnectorType> WEST = EnumProperty.<ConnectorType>create("west", ConnectorType.class);
    public static final EnumProperty<ConnectorType> EAST = EnumProperty.<ConnectorType>create("east", ConnectorType.class);
    public static final EnumProperty<ConnectorType> UP = EnumProperty.<ConnectorType>create("up", ConnectorType.class);
    public static final EnumProperty<ConnectorType> DOWN = EnumProperty.<ConnectorType>create("down", ConnectorType.class);

    public static final ModelProperty<BlockState> FACADEID = new ModelProperty<>();
  
```

下一部分是我们方块的形状，我们希望方块的模型和我们实际的电缆模型一致，这也是为什么当特定方向上是线缆的时候具有六个形状，当特定方向是方块时候有两个形状，因为我们的getShape（）必须是高效的， 所以我们有一个ShapeCache，存储了所有可能的形状。

 makeShapes（）负责创建缓存，他由构造函数调用。calculateShapeIndex（）函数根据六个方向上的连接类型计算缓存中的索引。makeShape（）是基于六个方向创建形状。combineShape（）函数将和特定的形状形成组合。如果电缆连接的是电缆我们只需要简单的显示电缆，如果连接的是块，则需要显示电缆和块连接的形状。

updateShape（）是当临近方块改变时候回调的方法，这种情况下，我们选哟重新计算形状。

```java

    private static VoxelShape[] shapeCache = null;

    private static final VoxelShape SHAPE_CABLE_NORTH = Shapes.box(.4, .4, 0, .6, .6, .4);
    private static final VoxelShape SHAPE_CABLE_SOUTH = Shapes.box(.4, .4, .6, .6, .6, 1);
    private static final VoxelShape SHAPE_CABLE_WEST = Shapes.box(0, .4, .4, .4, .6, .6);
    private static final VoxelShape SHAPE_CABLE_EAST = Shapes.box(.6, .4, .4, 1, .6, .6);
    private static final VoxelShape SHAPE_CABLE_UP = Shapes.box(.4, .6, .4, .6, 1, .6);
    private static final VoxelShape SHAPE_CABLE_DOWN = Shapes.box(.4, 0, .4, .6, .4, .6);

    private static final VoxelShape SHAPE_BLOCK_NORTH = Shapes.box(.2, .2, 0, .8, .8, .1);
    private static final VoxelShape SHAPE_BLOCK_SOUTH = Shapes.box(.2, .2, .9, .8, .8, 1);
    private static final VoxelShape SHAPE_BLOCK_WEST = Shapes.box(0, .2, .2, .1, .8, .8);
    private static final VoxelShape SHAPE_BLOCK_EAST = Shapes.box(.9, .2, .2, 1, .8, .8);
    private static final VoxelShape SHAPE_BLOCK_UP = Shapes.box(.2, .9, .2, .8, 1, .8);
    private static final VoxelShape SHAPE_BLOCK_DOWN = Shapes.box(.2, 0, .2, .8, .1, .8);

    private int calculateShapeIndex(ConnectorType north, ConnectorType south, ConnectorType west, ConnectorType east, ConnectorType up, ConnectorType down) {
        int l = ConnectorType.values().length;
        return ((((south.ordinal() * l + north.ordinal()) * l + west.ordinal()) * l + east.ordinal()) * l + up.ordinal()) * l + down.ordinal();
    }

    private void makeShapes() {
        if (shapeCache == null) {
            int length = ConnectorType.values().length;
            shapeCache = new VoxelShape[length * length * length * length * length * length];

            for (ConnectorType up : ConnectorType.VALUES) {
                for (ConnectorType down : ConnectorType.VALUES) {
                    for (ConnectorType north : ConnectorType.VALUES) {
                        for (ConnectorType south : ConnectorType.VALUES) {
                            for (ConnectorType east : ConnectorType.VALUES) {
                                for (ConnectorType west : ConnectorType.VALUES) {
                                    int idx = calculateShapeIndex(north, south, west, east, up, down);
                                    shapeCache[idx] = makeShape(north, south, west, east, up, down);
                                }
                            }
                        }
                    }
                }
            }

        }
    }

    private VoxelShape makeShape(ConnectorType north, ConnectorType south, ConnectorType west, ConnectorType east, ConnectorType up, ConnectorType down) {
        VoxelShape shape = Shapes.box(.4, .4, .4, .6, .6, .6);
        shape = combineShape(shape, north, SHAPE_CABLE_NORTH, SHAPE_BLOCK_NORTH);
        shape = combineShape(shape, south, SHAPE_CABLE_SOUTH, SHAPE_BLOCK_SOUTH);
        shape = combineShape(shape, west, SHAPE_CABLE_WEST, SHAPE_BLOCK_WEST);
        shape = combineShape(shape, east, SHAPE_CABLE_EAST, SHAPE_BLOCK_EAST);
        shape = combineShape(shape, up, SHAPE_CABLE_UP, SHAPE_BLOCK_UP);
        shape = combineShape(shape, down, SHAPE_CABLE_DOWN, SHAPE_BLOCK_DOWN);
        return shape;
    }
  
    private VoxelShape combineShape(VoxelShape shape, ConnectorType connectorType, VoxelShape cableShape, VoxelShape blockShape) {
        if (connectorType == ConnectorType.CABLE) {
            return Shapes.join(shape, cableShape, BooleanOp.OR);
        } else if (connectorType == ConnectorType.BLOCK) {
            return Shapes.join(shape, Shapes.join(blockShape, cableShape, BooleanOp.OR), BooleanOp.OR);
        } else {
            return shape;
        }
    }
  
    @Nonnull
    @Override
    public VoxelShape getShape(@Nonnull BlockState state, @Nonnull BlockGetter world, @Nonnull BlockPos pos, @Nonnull CollisionContext context) {
        ConnectorType north = getConnectorType(world, pos, Direction.NORTH);
        ConnectorType south = getConnectorType(world, pos, Direction.SOUTH);
        ConnectorType west = getConnectorType(world, pos, Direction.WEST);
        ConnectorType east = getConnectorType(world, pos, Direction.EAST);
        ConnectorType up = getConnectorType(world, pos, Direction.UP);
        ConnectorType down = getConnectorType(world, pos, Direction.DOWN);
        int index = calculateShapeIndex(north, south, west, east, up, down);
        return shapeCache[index];
    }

    @Nonnull
    @Override
    public BlockState updateShape(BlockState state, @Nonnull Direction direction, @Nonnull BlockState neighbourState, @Nonnull LevelAccessor world, @Nonnull BlockPos current, @Nonnull BlockPos offset) {
        if (state.getValue(WATERLOGGED)) {
            world.getFluidTicks().schedule(new ScheduledTick<>(Fluids.WATER, current, Fluids.WATER.getTickDelay(world), 0L));   // @todo 1.18 what is this last parameter exactly?
        }
        return calculateState(world, current, state);
    }
```

现在我们有了构造方法（在这里调用makeShapes（））我们设置含水方块是false，我们还需要为block entity 和block entity tiker实现相应的功能。

```java

    public CableBlock() {
        super(Properties.of()
                .strength(1.0f)
                .sound(SoundType.METAL)
                .noOcclusion()
        );
        makeShapes();
        registerDefaultState(defaultBlockState().setValue(WATERLOGGED, false));
    }

    @Nullable
    @Override
    public BlockEntity newBlockEntity(BlockPos blockPos, BlockState blockState) {
        return new CableBlockEntity(blockPos, blockState);
    }

    @Nullable
    @Override
    public <T extends BlockEntity> BlockEntityTicker<T> getTicker(Level level, BlockState state, BlockEntityType<T> type) {
        if (level.isClientSide) {
            return null;
        } else {
            return (lvl, pos, st, be) -> {
                if (be instanceof CableBlockEntity cable) {
                    cable.tickServer();
                }
            };
        }
    }
```

neighborChanged()和setPlacedBy用于标记实体的脏位，当某些东西改变的时候。这是为了让我们方块可以更新电缆网络（之后会介绍）

```java

    @Override
    public void neighborChanged(BlockState state, Level level, BlockPos pos, Block block, BlockPos fromPos, boolean isMoving) {
        super.neighborChanged(state, level, pos, block, fromPos, isMoving);
        if (!level.isClientSide && level.getBlockEntity(pos) instanceof CableBlockEntity cable) {
            cable.markDirty();
        }
    }

    @Override
    public void setPlacedBy(@Nonnull Level level, @Nonnull BlockPos pos, @Nonnull BlockState state, @Nullable LivingEntity placer, @Nonnull ItemStack stack) {
        super.setPlacedBy(level, pos, state, placer, stack);
        if (!level.isClientSide && level.getBlockEntity(pos) instanceof CableBlockEntity cable) {
            cable.markDirty();
        }
        BlockState blockState = calculateState(level, pos, state);
        if (state != blockState) {
            level.setBlockAndUpdate(pos, blockState);
        }
    }
```

getConnectorType()和isConnectable()方法用于确定在某个方向上的连接类型。这用于计算线缆的形状。

```java

    // Return the connector type for the given position and facing direction
    private ConnectorType getConnectorType(BlockGetter world, BlockPos connectorPos, Direction facing) {
        BlockPos pos = connectorPos.relative(facing);
        BlockState state = world.getBlockState(pos);
        Block block = state.getBlock();
        if (block instanceof CableBlock) {
            return ConnectorType.CABLE;
        } else if (isConnectable(world, connectorPos, facing)) {
            return ConnectorType.BLOCK;
        } else {
            return ConnectorType.NONE;
        }
    }

    // Return true if the block at the given position is connectable to a cable. This is the
    // case if the block supports forge energy
    public static boolean isConnectable(BlockGetter world, BlockPos connectorPos, Direction facing) {
        BlockPos pos = connectorPos.relative(facing);
        BlockState state = world.getBlockState(pos);
        if (state.isAir()) {
            return false;
        }
        BlockEntity te = world.getBlockEntity(pos);
        if (te == null) {
            return false;
        }
        return te.getCapability(ForgeCapabilities.ENERGY).isPresent();
    }
```

剩余的功能是定义和设置方块的状态，是必须要的。支持含水方块很简单，仅选哟台南佳WATERLOGGED属性，并重写getFluidState（）方法。

```java
    @Override
    protected void createBlockStateDefinition(@Nonnull StateDefinition.Builder<Block, BlockState> builder) {
        super.createBlockStateDefinition(builder);
        builder.add(WATERLOGGED, NORTH, SOUTH, EAST, WEST, UP, DOWN);
    }

    @Nullable
    @Override
    public BlockState getStateForPlacement(BlockPlaceContext context) {
        Level world = context.getLevel();
        BlockPos pos = context.getClickedPos();
        return calculateState(world, pos, defaultBlockState())
                .setValue(WATERLOGGED, world.getFluidState(pos).getType() == Fluids.WATER);
    }

    @Nonnull
    private BlockState calculateState(LevelAccessor world, BlockPos pos, BlockState state) {
        ConnectorType north = getConnectorType(world, pos, Direction.NORTH);
        ConnectorType south = getConnectorType(world, pos, Direction.SOUTH);
        ConnectorType west = getConnectorType(world, pos, Direction.WEST);
        ConnectorType east = getConnectorType(world, pos, Direction.EAST);
        ConnectorType up = getConnectorType(world, pos, Direction.UP);
        ConnectorType down = getConnectorType(world, pos, Direction.DOWN);

        return state
                .setValue(NORTH, north)
                .setValue(SOUTH, south)
                .setValue(WEST, west)
                .setValue(EAST, east)
                .setValue(UP, up)
                .setValue(DOWN, down);
    }

    @Nonnull
    @Override
    public FluidState getFluidState(BlockState state) {
        return state.getValue(WATERLOGGED) ? Fluids.WATER.getSource(false) : super.getFluidState(state);
    }
}
```

#### Cable Block Entity

Cable Block Entity负责追踪线缆网络。还负责追踪流过线缆的电量，电缆的网络简单的由一组具有能量接收器的坐标表示。当任意一个相邻的方块改变时候都需要重新计算网络。当方块放置或者移除的时候也需要重新计算网络。

- WARN：这里给出的网络的实现是可行的，但是并不完美。这只是一个简单的实现，对我们的目的有效。更先进的mod（例如XNet）具有更先进的有线网络，并将其网络数据缓存在SavedData结构中。

这个block entity的第一部分和之前的一样，因为线缆也是一个energy handler，所以需要这种capability。

```java

public class CableBlockEntity extends BlockEntity {

    public static final String ENERGY_TAG = "Energy";

    public static final int MAXTRANSFER = 100;
    public static final int CAPACITY = 1000;

    private final EnergyStorage energy = createEnergyStorage();
    private final LazyOptional<IEnergyStorage> energyHandler = LazyOptional.of(() -> new AdaptedEnergyStorage(energy) {
        @Override
        public int extractEnergy(int maxExtract, boolean simulate) {
            return 0;
        }

        @Override
        public int receiveEnergy(int maxReceive, boolean simulate) {
            setChanged();
            return super.receiveEnergy(maxReceive, simulate);
        }

        @Override
        public boolean canExtract() {
            return false;
        }

        @Override
        public boolean canReceive() {
            return true;
        }
    });

    protected CableBlockEntity(BlockEntityType<?> type, BlockPos pos, BlockState state) {
        super(type, pos, state);
    }

    public CableBlockEntity(BlockPos pos, BlockState state) {
        super(Registration.CABLE_BLOCK_ENTITY.get(), pos, state);
    }
  
```

以下的block负责缓存输出，对于连接到网络的所有能量接收者，他们所接受的能量是一个懒惰计算集。checkOutputs（）函数计算此合集。他将遍历连接到该电缆，然后检查该线缆连接的所有能量接收器。markDirty（）函数将使得当前电缆和所有连接的电缆的输出缓存无效化。当电缆网络发生变化的时候需要这样做。

traverse（）方法是一个通用的方法，他将会遍历所有连接到此电缆的电缆并调用他们的comsumer

```java

    // Cached outputs
    private Set<BlockPos> outputs = null;

    // This function will cache all outputs for this cable network. It will do this
    // by traversing all cables connected to this cable and then check for all energy
    // receivers around those cables.
    private void checkOutputs() {
        if (outputs == null) {
            outputs = new HashSet<>();
            traverse(worldPosition, cable -> {
                // Check for all energy receivers around this position (ignore cables)
                for (Direction direction : Direction.values()) {
                    BlockPos p = cable.getBlockPos().relative(direction);
                    BlockEntity te = level.getBlockEntity(p);
                    if (te != null && !(te instanceof CableBlockEntity)) {
                        te.getCapability(ForgeCapabilities.ENERGY).ifPresent(handler -> {
                            if (handler.canReceive()) {
                                outputs.add(p);
                            }
                        });
                    }
                }
            });
        }
    }

    public void markDirty() {
        traverse(worldPosition, cable -> cable.outputs = null);
    }

    // This is a generic function that will traverse all cables connected to this cable
    // and call the given consumer for each cable.
    private void traverse(BlockPos pos, Consumer<CableBlockEntity> consumer) {
        Set<BlockPos> traversed = new HashSet<>();
        traversed.add(pos);
        consumer.accept(this);
        traverse(pos, traversed, consumer);
    }

    private void traverse(BlockPos pos, Set<BlockPos> traversed, Consumer<CableBlockEntity> consumer) {
        for (Direction direction : Direction.values()) {
            BlockPos p = pos.relative(direction);
            if (!traversed.contains(p)) {
                traversed.add(p);
                if (level.getBlockEntity(p) instanceof CableBlockEntity cable) {
                    consumer.accept(cable);
                    cable.traverse(p, traversed, consumer);
                }
            }
        }
    }

```

tickServer()方法在服务器上的每个tick都会回调。他将会将能量分发给所有的outputs。首先他会检查电缆中是否存在能量，如果没有能量，那么我们什么也不需要做。如果由能量，之后我们需要检查是否存在输出，如果没有任何输出，我们什么也不需要做。如果存在输出，那么我们分发能量到每一个outputs上。我们通过将能量除于所有的outpus来实现，然后对于每一个output我们检查它是否可以接受能量，如果可以接受我们就将能量发送给这个output。我们通过获得output的energy capability调用receiveEnergy（）方法，这个方法将会返回机器所接受的能量，我们之后减去线缆中对应的能量。

- WARN：同样，这不是一个完美的算法，按照这样的实现方式，一些接收器接受到能量可能会比其他的少。这是因为我们将能量除以了所有的outputs，然后一个一个的将能量发送给他们，如果第一个output不能接受能量，我们需要将能量发送给第二个output。如果第二个output可以接受能量，之后它将获得所有能量。如果第二个output不能接受能量，我们需要将能量发送给第三个output，以此类推。这意味第一个output获得的能量比第二个少。这对于我们的目的并不是一个问题，但是我们仍需要了解。

```java

    public void tickServer() {
        if (energy.getEnergyStored() > 0) {
            // Only do something if we have energy
            checkOutputs();
            if (!outputs.isEmpty()) {
                // Distribute energy over all outputs
                int amount = energy.getEnergyStored() / outputs.size();
                for (BlockPos p : outputs) {
                    BlockEntity te = level.getBlockEntity(p);
                    if (te != null) {
                        te.getCapability(ForgeCapabilities.ENERGY).ifPresent(handler -> {
                            if (handler.canReceive()) {
                                int received = handler.receiveEnergy(amount, false);
                                energy.extractEnergy(received, false);
                            }
                        });
                    }
                }
            }
        }
    }

    @Override
    protected void saveAdditional(CompoundTag tag) {
        super.saveAdditional(tag);
        tag.put(ENERGY_TAG, energy.serializeNBT());
    }

    @Override
    public void load(CompoundTag tag) {
        super.load(tag);
        if (tag.contains(ENERGY_TAG)) {
            energy.deserializeNBT(tag.get(ENERGY_TAG));
        }
    }

    @Nonnull
    private EnergyStorage createEnergyStorage() {
        return new EnergyStorage(CAPACITY, MAXTRANSFER, MAXTRANSFER);
    }

    @NotNull
    @Override
    public <T> LazyOptional<T> getCapability(@NotNull Capability<T> cap, @Nullable Direction side) {
        if (cap == ForgeCapabilities.ENERGY) {
            return energyHandler.cast();
        } else {
            return super.getCapability(cap, side);
        }
    }
}

```

### The Facade

The Facade 是一个 block，它可以用于模仿另一个方块，facade 实际上是一个特殊的电缆。这意味着FacadeBlock应该继承CableBlock，同样FacadeBlockEntity继承CableBlockEntity，让我们复习下代码：

#### The Facade Block

Facade Block和电缆方块类似，除此之外还有一些逻辑，当facade方块被破坏，应该回复到原本的线缆，此外我们还需要覆盖getShape方法，以便返回模拟块的形状。

```java

public class FacadeBlock extends CableBlock implements EntityBlock {

    public FacadeBlock() {
        super();
    }

    @Nullable
    @Override
    public BlockEntity newBlockEntity(@NotNull BlockPos pos, @NotNull BlockState state) {
        return new FacadeBlockEntity(pos, state);
    }

    @NotNull
    @Override
    public VoxelShape getShape(@NotNull BlockState state, @NotNull BlockGetter world, @NotNull BlockPos pos, @NotNull CollisionContext context) {
        if (world.getBlockEntity(pos) instanceof FacadeBlockEntity facade) {
            BlockState mimicBlock = facade.getMimicBlock();
            if (mimicBlock != null) {
                return mimicBlock.getShape(world, pos, context);
            }
        }
        return super.getShape(state, world, pos, context);
    }

    // This function is called when the facade block is succesfully harvested by the player
    // When the player destroys the facade we need to drop the facade block item with the correct mimiced block
    @Override
    public void playerDestroy(@Nonnull Level level, @Nonnull Player player, @Nonnull BlockPos pos, @Nonnull BlockState state, @Nullable BlockEntity te, @Nonnull ItemStack stack) {
        ItemStack item = new ItemStack(Registration.FACADE_BLOCK.get());
        BlockState mimicBlock;
        if (te instanceof FacadeBlockEntity) {
            mimicBlock = ((FacadeBlockEntity) te).getMimicBlock();
        } else {
            mimicBlock = Blocks.COBBLESTONE.defaultBlockState();
        }
        FacadeBlockItem.setMimicBlock(item, mimicBlock);
        popResource(level, pos, item);
    }

    // When the player destroys the facade we need to restore the cable block
    @Override
    public boolean onDestroyedByPlayer(BlockState state, Level world, BlockPos pos, Player player, boolean willHarvest, FluidState fluid) {
        BlockState defaultState = Registration.CABLE_BLOCK.get().defaultBlockState();
        BlockState newState = CableBlock.calculateState(world, pos, defaultState);
        return ((LevelAccessor) world).setBlock(pos, newState, ((LevelAccessor) world).isClientSide()
                ? Block.UPDATE_ALL + Block.UPDATE_IMMEDIATE
                : Block.UPDATE_ALL);
    }

}
```

#### The Facade Block Entity

Facade Block Entity类似线缆的block entity，不过Facade Block Entity 还需要追踪模仿的方块，它应该拓展于CableBlockEntity，所以它也需要被识别为传输电力的有效方块。

需要值得注意的是烘焙模型不能访问level，因为没法访问方块实体。这意味着我们不能通过方块实体获得模拟的方块，相反，我们需要通过模型数据系统传达信息。

查看代码中的注释，了解每个方法详细是做什么的

```java
public class FacadeBlockEntity extends CableBlockEntity {

    public static final String MIMIC_TAG = "mimic";

    @Nullable private BlockState mimicBlock = null;

    public FacadeBlockEntity(BlockPos pos, BlockState state) {
        super(Registration.FACADE_BLOCK_ENTITY.get(), pos, state);
    }

    // The default onDataPacket() will call load() to load the data from the packet.
    // In addition to that we send a block update to the client
    // and also request a model data update (for the cable baked model)
    @Override
    public void onDataPacket(Connection net, ClientboundBlockEntityDataPacket packet) {
        super.onDataPacket(net, packet);

        if (level.isClientSide) {
            level.sendBlockUpdated(worldPosition, getBlockState(), getBlockState(), Block.UPDATE_ALL);
            requestModelDataUpdate();
        }
    }

    // getUpdatePacket() is called on the server when a block is placed or updated.
    // It should return a packet containing all information needed to render this block on the client.
    // In our case this is the block mimic information. On the client side onDataPacket() is called
    // with this packet.
    @Nullable
    @Override
    public ClientboundBlockEntityDataPacket getUpdatePacket() {
        CompoundTag nbtTag = new CompoundTag();
        saveMimic(nbtTag);
        return ClientboundBlockEntityDataPacket.create(this, (BlockEntity entity) -> {return nbtTag;});
    }

    // getUpdateTag() is called on the server on initial load of the chunk. It will cause
    // the packet to be sent to the client and handleUpdateTag() will be called on the client.
    // The default implementation of handleUpdateTag() will call load() to load the data from the packet.
    // In our case this is sufficient
    @Nonnull
    @Override
    public CompoundTag getUpdateTag() {
        CompoundTag updateTag = super.getUpdateTag();
        saveMimic(updateTag);
        return updateTag;
    }

    public @Nullable BlockState getMimicBlock() {
        return mimicBlock;
    }

    // This is used to build the model data for the cable baked model.
    @Nonnull
    @Override
    public ModelData getModelData() {
        return ModelData.builder()
                .with(CableBlock.FACADEID, mimicBlock)
                .build();
    }


    public void setMimicBlock(BlockState mimicBlock) {
        this.mimicBlock = mimicBlock;
        setChanged();
        getLevel().sendBlockUpdated(getBlockPos(), getBlockState(), getBlockState(), Block.UPDATE_CLIENTS + Block.UPDATE_NEIGHBORS);
    }

    @Override
    public void load(CompoundTag tagCompound) {
        super.load(tagCompound);
        loadMimic(tagCompound);
    }

    private void loadMimic(CompoundTag tagCompound) {
        if (tagCompound.contains(MIMIC_TAG)) {
            mimicBlock = NbtUtils.readBlockState(BuiltInRegistries.BLOCK.asLookup(), tagCompound.getCompound(MIMIC_TAG));
        } else {
            mimicBlock = null;
        }
    }

    @Override
    public void saveAdditional(@Nonnull CompoundTag tagCompound) {
        super.saveAdditional(tagCompound);
        saveMimic(tagCompound);
    }

    private void saveMimic(@NotNull CompoundTag tagCompound) {
        if (mimicBlock != null) {
            CompoundTag tag = NbtUtils.writeBlockState(mimicBlock);
            tagCompound.put(MIMIC_TAG, tag);
        }
    }
}
```

#### The Facade Block Item

由于当facade Block放置的时候我们需要一些特殊的处理，我们需要为他创建一个自定义的block item，FacadeBlockItme，负责放置Facade时候设置模拟方块。

```java
public class FacadeBlockItem extends BlockItem {

    public static final String FACADE_IS_MIMICING = "tutorial.facade.is_mimicing";

    private static String getMimickingString(ItemStack stack) {
        CompoundTag tag = stack.getTag();
        if (tag != null) {
            CompoundTag mimic = tag.getCompound("mimic");
            Block value = ForgeRegistries.BLOCKS.getValue(new ResourceLocation(mimic.getString("Name")));
            if (value != null) {
                ItemStack s = new ItemStack(value, 1);
                s.getItem();
                return s.getHoverName().getString();
            }
        }
        return "<unset>";
    }


    public FacadeBlockItem(FacadeBlock block, Item.Properties properties) {
        super(block, properties);
    }

    private static void userSetMimicBlock(@Nonnull ItemStack item, BlockState mimicBlock, UseOnContext context) {
        Level world = context.getLevel();
        Player player = context.getPlayer();
        setMimicBlock(item, mimicBlock);
        if (world.isClientSide) {
            player.displayClientMessage(Component.translatable(FACADE_IS_MIMICING, mimicBlock.getBlock().getDescriptionId()), false);
        }
    }

    public static void setMimicBlock(@Nonnull ItemStack item, BlockState mimicBlock) {
        CompoundTag tagCompound = new CompoundTag();
        CompoundTag nbt = NbtUtils.writeBlockState(mimicBlock);
        tagCompound.put("mimic", nbt);
        item.setTag(tagCompound);
    }

    public static BlockState getMimicBlock(Level level, @Nonnull ItemStack stack) {
        CompoundTag tagCompound = stack.getTag();
        if (tagCompound == null || !tagCompound.contains("mimic")) {
            return Blocks.COBBLESTONE.defaultBlockState();
        } else {
            return NbtUtils.readBlockState(BuiltInRegistries.BLOCK.asLookup(), tagCompound.getCompound("mimic"));
        }
    }

    @Override
    protected boolean canPlace(@Nonnull BlockPlaceContext context, @Nonnull BlockState state) {
        return true;
    }

    // This function is called when our block item is right clicked on something. When this happens
    // we want to either set the minic block or place the facade block
    @Nonnull
    @Override
    public InteractionResult useOn(UseOnContext context) {
        Level world = context.getLevel();
        BlockPos pos = context.getClickedPos();
        Player player = context.getPlayer();
        BlockState state = world.getBlockState(pos);
        Block block = state.getBlock();

        ItemStack itemstack = context.getItemInHand();

        if (!itemstack.isEmpty()) {

            if (block == Registration.CABLE_BLOCK.get()) {
                // We are hitting a cable block. We want to replace it with a facade block
                FacadeBlock facadeBlock = (FacadeBlock) this.getBlock();
                BlockPlaceContext blockContext = new ReplaceBlockItemUseContext(context);
                BlockState placementState = facadeBlock.getStateForPlacement(blockContext)
                        .setValue(NORTH, state.getValue(NORTH))
                        .setValue(SOUTH, state.getValue(SOUTH))
                        .setValue(WEST, state.getValue(WEST))
                        .setValue(EAST, state.getValue(EAST))
                        .setValue(UP, state.getValue(UP))
                        .setValue(DOWN, state.getValue(DOWN))
                        ;

                if (placeBlock(blockContext, placementState)) {
                    SoundType soundtype = world.getBlockState(pos).getBlock().getSoundType(world.getBlockState(pos), world, pos, player);
                    world.playSound(player, pos, soundtype.getPlaceSound(), SoundSource.BLOCKS, (soundtype.getVolume() + 1.0F) / 2.0F, soundtype.getPitch() * 0.8F);
                    BlockEntity te = world.getBlockEntity(pos);
                    if (te instanceof FacadeBlockEntity) {
                        ((FacadeBlockEntity) te).setMimicBlock(getMimicBlock(world, itemstack));
                    }
                    int amount = -1;
                    itemstack.grow(amount);
                }
            } else if (block == Registration.FACADE_BLOCK.get()) {
                // We are hitting a facade block. We want to copy the block it is mimicing
                BlockEntity te = world.getBlockEntity(pos);
                if (!(te instanceof FacadeBlockEntity facade)) {
                    return InteractionResult.FAIL;
                }
                if (facade.getMimicBlock() == null) {
                    return InteractionResult.FAIL;
                }
                userSetMimicBlock(itemstack, facade.getMimicBlock(), context);
            } else {
                // We are hitting something else. We want to set that block as what we are going to mimic
                userSetMimicBlock(itemstack, state, context);
            }
            return InteractionResult.SUCCESS;
        } else {
            return InteractionResult.FAIL;
        }
    }

    @Override
    public void appendHoverText(@Nonnull ItemStack stack, @Nullable Level level, @Nonnull List<Component> tooltip, @Nonnull TooltipFlag flag) {
        super.appendHoverText(stack, level, tooltip, flag);
        if (stack.hasTag()) {
            tooltip.add(Component.translatable(FACADE_IS_MIMICING, getMimickingString(stack)));
        }
    }
}
```

我们需要一个类帮助我们进行右键的处理，ReplaceBlockItemUseContext，BlockPlaceContext会将replaceClicked设置为True，这将保证我们的facade放置的时候会替换cable。

```java
public class ReplaceBlockItemUseContext extends BlockPlaceContext {

    public ReplaceBlockItemUseContext(UseOnContext context) {
        super(context);
        replaceClicked = true;
    }
}
```



### 注册Registration

在这里注册我们的cable  和 facade方块

```java


public class Registration {
    
    ...
    
    public static final RegistryObject<CableBlock> CABLE_BLOCK = BLOCKS.register("cable", CableBlock::new);
    public static final RegistryObject<Item> CABLE_BLOCK_ITEM = ITEMS.register("cable", () -> new BlockItem(CABLE_BLOCK.get(), new Item.Properties()));
    public static final RegistryObject<BlockEntityType<CableBlockEntity>> CABLE_BLOCK_ENTITY = BLOCK_ENTITIES.register("cable",
            () -> BlockEntityType.Builder.of(CableBlockEntity::new, CABLE_BLOCK.get()).build(null));

    public static final RegistryObject<FacadeBlock> FACADE_BLOCK = BLOCKS.register("facade", FacadeBlock::new);
    public static final RegistryObject<Item> FACADE_BLOCK_ITEM = ITEMS.register("facade", () -> new FacadeBlockItem(FACADE_BLOCK.get(), new Item.Properties()));
    public static final RegistryObject<BlockEntityType<FacadeBlockEntity>> FACADE_BLOCK_ENTITY = BLOCK_ENTITIES.register("facade",
            () -> BlockEntityType.Builder.of(FacadeBlockEntity::new, FACADE_BLOCK.get()).build(null));

    public static RegistryObject<CreativeModeTab> TAB = TABS.register("tutpower", () -> CreativeModeTab.builder()
            .title(Component.translatable("tab.tutpower"))
            .icon(() -> new ItemStack(GENERATOR_BLOCK.get()))
            .withTabsBefore(CreativeModeTabs.SPAWN_EGGS)
            .displayItems((featureFlags, output) -> {
                output.accept(GENERATOR_BLOCK.get());
                output.accept(CHARGER_BLOCK.get());
                output.accept(CABLE_BLOCK.get());
                output.accept(FACADE_BLOCK.get());
            })
            .build());
}
```



### 烘焙模型Baked Model

烘焙模型负责生成电缆的实际的模型，通过查看六个方向和六个方向上的类型然后生成适当的立体方块实现此目的，电缆方块和facade方块都使用相同的烘焙模型

#### 烘焙模型加载器The Baked Model Loader

要实现烘焙模型，你首先需要实现模型加载器。该加载器负责从json中加载模型，在我们的例子中，我们有一个json文件，用于电缆方块和facade方块，因此我们需要区分两者，我们在接送文件中添加facade属性。对于facade方块，该属性为true，对于电缆方块，该属性为fasle，加载器读取此属性，然后创建适当的CableBakeModel

```java


public class CableModelLoader implements IGeometryLoader<CableModelLoader.CableModelGeometry> {

    public static final ResourceLocation GENERATOR_LOADER = new ResourceLocation(TutorialPower.MODID, "cableloader");

    public static void register(ModelEvent.RegisterGeometryLoaders event) {
        event.register("cableloader", new CableModelLoader());
    }


    @Override
    public CableModelGeometry read(JsonObject jsonObject, JsonDeserializationContext deserializationContext) throws JsonParseException {
        boolean facade = jsonObject.has("facade") && jsonObject.get("facade").getAsBoolean();
        return new CableModelGeometry(facade);
    }

    public static class CableModelGeometry implements IUnbakedGeometry<CableModelGeometry> {

        private final boolean facade;

        public CableModelGeometry(boolean facade) {
            this.facade = facade;
        }

        @Override
        public BakedModel bake(IGeometryBakingContext context, ModelBaker baker, Function<Material, TextureAtlasSprite> spriteGetter, ModelState modelState, ItemOverrides overrides, ResourceLocation modelLocation) {
            return new CableBakedModel(context, facade);
        }
    }
}
```

`register()`方法需要在`ModelEvent.RegisterGeometryLoaders`事件中调用，我们在`ClientSetup`中做到这一点，我们还需要注册方块颜色处理器，之后会介绍，这个颜色处理器确保我们在模仿类似草方块时候可以正确的从生物群系中获得颜色。

```java
@Mod.EventBusSubscriber(modid = MODID, bus = Mod.EventBusSubscriber.Bus.MOD, value = Dist.CLIENT)
public class ClientSetup {

    ...

    @SubscribeEvent
    public static void modelInit(ModelEvent.RegisterGeometryLoaders event) {
        CableModelLoader.register(event);
    }

    @SubscribeEvent
    public static void registerBlockColor(RegisterColorHandlersEvent.Block event) {
        event.register(new FacadeBlockColor(), Registration.FACADE_BLOCK.get());
    }

}
```

#### 方块颜色处理器The block color handler

当我们模仿另一个方块的时候需要确保方块的颜色是正确的，例如，我们模仿草方块，需要确保草方块颜色和当前的生物群系颜色是一致的。

```java

public class FacadeBlockColor implements BlockColor {

    @Override
    public int getColor(@Nonnull BlockState blockState, @Nullable BlockAndTintGetter world, @Nullable BlockPos pos, int tint) {
        if (world != null) {
            BlockEntity te = world.getBlockEntity(pos);
            if (te instanceof FacadeBlockEntity facade) {
                BlockState mimic = facade.getMimicBlock();
                if (mimic != null) {
                    return Minecraft.getInstance().getBlockColors().getColor(mimic, world, pos, tint);
                }
            }
        }
        return -1;
    }
}
```

#### 烘焙模型The Baked Model

烘焙模型负责生成电缆的实际模型。它通过六个方向和六个方向行上的线缆类型来生成对应的四边形，电缆方块和facade方块使用相同的烘焙模型。

代码使用CablePatterns辅助类生成四边形，该类知道如何将特定的连接器转为正确的四边形。

这个类最总要的就是线程就是getQuads（）线程，该线程被渲染器调用获得线缆的四边形。

```java

public class CableBakedModel implements IDynamicBakedModel {

    private final IGeometryBakingContext context;
    private final boolean facade;

    private TextureAtlasSprite spriteConnector;
    private TextureAtlasSprite spriteNoneCable;
    private TextureAtlasSprite spriteNormalCable;
    private TextureAtlasSprite spriteEndCable;
    private TextureAtlasSprite spriteCornerCable;
    private TextureAtlasSprite spriteThreeCable;
    private TextureAtlasSprite spriteCrossCable;
    private TextureAtlasSprite spriteSide;

    static {
        // For all possible patterns we define the sprite to use and the rotation. Note that each
        // pattern looks at the existance of a cable section for each of the four directions
        // excluding the one we are looking at.
        CablePatterns.PATTERNS.put(Pattern.of(false, false, false, false), QuadSetting.of(SPRITE_NONE, 0));
        CablePatterns.PATTERNS.put(Pattern.of(true, false, false, false), QuadSetting.of(SPRITE_END, 3));
        CablePatterns.PATTERNS.put(Pattern.of(false, true, false, false), QuadSetting.of(SPRITE_END, 0));
        CablePatterns.PATTERNS.put(Pattern.of(false, false, true, false), QuadSetting.of(SPRITE_END, 1));
        CablePatterns.PATTERNS.put(Pattern.of(false, false, false, true), QuadSetting.of(SPRITE_END, 2));
        CablePatterns.PATTERNS.put(Pattern.of(true, true, false, false), QuadSetting.of(SPRITE_CORNER, 0));
        CablePatterns.PATTERNS.put(Pattern.of(false, true, true, false), QuadSetting.of(SPRITE_CORNER, 1));
        CablePatterns.PATTERNS.put(Pattern.of(false, false, true, true), QuadSetting.of(SPRITE_CORNER, 2));
        CablePatterns.PATTERNS.put(Pattern.of(true, false, false, true), QuadSetting.of(SPRITE_CORNER, 3));
        CablePatterns.PATTERNS.put(Pattern.of(false, true, false, true), QuadSetting.of(SPRITE_STRAIGHT, 0));
        CablePatterns.PATTERNS.put(Pattern.of(true, false, true, false), QuadSetting.of(SPRITE_STRAIGHT, 1));
        CablePatterns.PATTERNS.put(Pattern.of(true, true, true, false), QuadSetting.of(SPRITE_THREE, 0));
        CablePatterns.PATTERNS.put(Pattern.of(false, true, true, true), QuadSetting.of(SPRITE_THREE, 1));
        CablePatterns.PATTERNS.put(Pattern.of(true, false, true, true), QuadSetting.of(SPRITE_THREE, 2));
        CablePatterns.PATTERNS.put(Pattern.of(true, true, false, true), QuadSetting.of(SPRITE_THREE, 3));
        CablePatterns.PATTERNS.put(Pattern.of(true, true, true, true), QuadSetting.of(SPRITE_CROSS, 0));
    }

    public CableBakedModel(IGeometryBakingContext context, boolean facade) {
        this.context = context;
        this.facade = facade;
    }

    private void initTextures() {
        if (spriteConnector == null) {
            spriteConnector = getTexture("block/cable/connector");
            spriteNormalCable = getTexture("block/cable/normal");
            spriteNoneCable = getTexture("block/cable/none");
            spriteEndCable = getTexture("block/cable/end");
            spriteCornerCable = getTexture("block/cable/corner");
            spriteThreeCable = getTexture("block/cable/three");
            spriteCrossCable = getTexture("block/cable/cross");
            spriteSide = getTexture("block/cable/side");
        }
    }

    // All textures are baked on a big texture atlas. This function gets the texture from that atlas
    private TextureAtlasSprite getTexture(String path) {
        return Minecraft.getInstance().getTextureAtlas(InventoryMenu.BLOCK_ATLAS).apply(new ResourceLocation(TutorialPower.MODID, path));
    }

    private TextureAtlasSprite getSpriteNormal(CablePatterns.SpriteIdx idx) {
        initTextures();
        return switch (idx) {
            case SPRITE_NONE -> spriteNoneCable;
            case SPRITE_END -> spriteEndCable;
            case SPRITE_STRAIGHT -> spriteNormalCable;
            case SPRITE_CORNER -> spriteCornerCable;
            case SPRITE_THREE -> spriteThreeCable;
            case SPRITE_CROSS -> spriteCrossCable;
        };
    }

    @Override
    public boolean usesBlockLight() {
        return false;
    }

    @Override
    @NotNull
    public List<BakedQuad> getQuads(@Nullable BlockState state, @Nullable Direction side, @NotNull RandomSource rand, @NotNull ModelData extraData, @Nullable RenderType layer) {
        initTextures();
        List<BakedQuad> quads = new ArrayList<>();
        if (side == null && (layer == null || layer.equals(RenderType.solid()))) {
            // Called with the blockstate from our block. Here we get the values of the six properties and pass that to
            // our baked model implementation. If state == null we are called from the inventory and we use the default
            // values for the properties
            ConnectorType north, south, west, east, up, down;
            if (state != null) {
                north = state.getValue(CableBlock.NORTH);
                south = state.getValue(CableBlock.SOUTH);
                west = state.getValue(CableBlock.WEST);
                east = state.getValue(CableBlock.EAST);
                up = state.getValue(CableBlock.UP);
                down = state.getValue(CableBlock.DOWN);
            } else {
                // If we are a facade and we are an item then we render as the 'side' texture as a full block
                if (facade) {
                    quads.add(quad(v(0, 1, 1), v(1, 1, 1), v(1, 1, 0), v(0, 1, 0), spriteSide));
                    quads.add(quad(v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1), spriteSide));
                    quads.add(quad(v(1, 0, 0), v(1, 1, 0), v(1, 1, 1), v(1, 0, 1), spriteSide));
                    quads.add(quad(v(0, 0, 1), v(0, 1, 1), v(0, 1, 0), v(0, 0, 0), spriteSide));
                    quads.add(quad(v(0, 1, 0), v(1, 1, 0), v(1, 0, 0), v(0, 0, 0), spriteSide));
                    quads.add(quad(v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1), spriteSide));
                    return quads;
                }
                north = south = west = east = up = down = NONE;
            }

            TextureAtlasSprite spriteCable = spriteNormalCable;
            Function<CablePatterns.SpriteIdx, TextureAtlasSprite> spriteGetter = this::getSpriteNormal;

            double o = .4;      // Thickness of the cable. .0 would be full block, .5 is infinitely thin.
            double p = .1;      // Thickness of the connector as it is put on the connecting block
            double q = .2;      // The wideness of the connector

            // For each side we either cap it off if there is no similar block adjacent on that side
            // or else we extend so that we touch the adjacent block:
            if (up == CABLE) {
                quads.add(quad(v(1 - o, 1, o), v(1 - o, 1, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1 - o, o), spriteCable));
                quads.add(quad(v(o, 1, 1 - o), v(o, 1, o), v(o, 1 - o, o), v(o, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(o, 1, o), v(1 - o, 1, o), v(1 - o, 1 - o, o), v(o, 1 - o, o), spriteCable));
                quads.add(quad(v(o, 1 - o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1, 1 - o), v(o, 1, 1 - o), spriteCable));
            } else if (up == BLOCK) {
                quads.add(quad(v(1 - o, 1 - p, o), v(1 - o, 1 - p, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1 - o, o), spriteCable));
                quads.add(quad(v(o, 1 - p, 1 - o), v(o, 1 - p, o), v(o, 1 - o, o), v(o, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(o, 1 - p, o), v(1 - o, 1 - p, o), v(1 - o, 1 - o, o), v(o, 1 - o, o), spriteCable));
                quads.add(quad(v(o, 1 - o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1 - p, 1 - o), v(o, 1 - p, 1 - o), spriteCable));

                quads.add(quad(v(1 - q, 1 - p, q), v(1 - q, 1, q), v(1 - q, 1, 1 - q), v(1 - q, 1 - p, 1 - q), spriteSide));
                quads.add(quad(v(q, 1 - p, 1 - q), v(q, 1, 1 - q), v(q, 1, q), v(q, 1 - p, q), spriteSide));
                quads.add(quad(v(q, 1, q), v(1 - q, 1, q), v(1 - q, 1 - p, q), v(q, 1 - p, q), spriteSide));
                quads.add(quad(v(q, 1 - p, 1 - q), v(1 - q, 1 - p, 1 - q), v(1 - q, 1, 1 - q), v(q, 1, 1 - q), spriteSide));

                quads.add(quad(v(q, 1 - p, q), v(1 - q, 1 - p, q), v(1 - q, 1 - p, 1 - q), v(q, 1 - p, 1 - q), spriteConnector));
                quads.add(quad(v(q, 1, q), v(q, 1, 1 - q), v(1 - q, 1, 1 - q), v(1 - q, 1, q), spriteSide));
            } else {
                QuadSetting pattern = CablePatterns.findPattern(west, south, east, north);
                quads.add(quad(v(o, 1 - o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1 - o, o), v(o, 1 - o, o), spriteGetter.apply(pattern.sprite()), pattern.rotation()));
            }

            if (down == CABLE) {
                quads.add(quad(v(1 - o, o, o), v(1 - o, o, 1 - o), v(1 - o, 0, 1 - o), v(1 - o, 0, o), spriteCable));
                quads.add(quad(v(o, o, 1 - o), v(o, o, o), v(o, 0, o), v(o, 0, 1 - o), spriteCable));
                quads.add(quad(v(o, o, o), v(1 - o, o, o), v(1 - o, 0, o), v(o, 0, o), spriteCable));
                quads.add(quad(v(o, 0, 1 - o), v(1 - o, 0, 1 - o), v(1 - o, o, 1 - o), v(o, o, 1 - o), spriteCable));
            } else if (down == BLOCK) {
                quads.add(quad(v(1 - o, o, o), v(1 - o, o, 1 - o), v(1 - o, p, 1 - o), v(1 - o, p, o), spriteCable));
                quads.add(quad(v(o, o, 1 - o), v(o, o, o), v(o, p, o), v(o, p, 1 - o), spriteCable));
                quads.add(quad(v(o, o, o), v(1 - o, o, o), v(1 - o, p, o), v(o, p, o), spriteCable));
                quads.add(quad(v(o, p, 1 - o), v(1 - o, p, 1 - o), v(1 - o, o, 1 - o), v(o, o, 1 - o), spriteCable));

                quads.add(quad(v(1 - q, 0, q), v(1 - q, p, q), v(1 - q, p, 1 - q), v(1 - q, 0, 1 - q), spriteSide));
                quads.add(quad(v(q, 0, 1 - q), v(q, p, 1 - q), v(q, p, q), v(q, 0, q), spriteSide));
                quads.add(quad(v(q, p, q), v(1 - q, p, q), v(1 - q, 0, q), v(q, 0, q), spriteSide));
                quads.add(quad(v(q, 0, 1 - q), v(1 - q, 0, 1 - q), v(1 - q, p, 1 - q), v(q, p, 1 - q), spriteSide));

                quads.add(quad(v(q, p, 1 - q), v(1 - q, p, 1 - q), v(1 - q, p, q), v(q, p, q), spriteConnector));
                quads.add(quad(v(q, 0, 1 - q), v(q, 0, q), v(1 - q, 0, q), v(1 - q, 0, 1 - q), spriteSide));
            } else {
                QuadSetting pattern = CablePatterns.findPattern(west, north, east, south);
                quads.add(quad(v(o, o, o), v(1 - o, o, o), v(1 - o, o, 1 - o), v(o, o, 1 - o), spriteGetter.apply(pattern.sprite()), pattern.rotation()));
            }

            if (east == CABLE) {
                quads.add(quad(v(1, 1 - o, 1 - o), v(1, 1 - o, o), v(1 - o, 1 - o, o), v(1 - o, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(1, o, o), v(1, o, 1 - o), v(1 - o, o, 1 - o), v(1 - o, o, o), spriteCable));
                quads.add(quad(v(1, 1 - o, o), v(1, o, o), v(1 - o, o, o), v(1 - o, 1 - o, o), spriteCable));
                quads.add(quad(v(1, o, 1 - o), v(1, 1 - o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, o, 1 - o), spriteCable));
            } else if (east == BLOCK) {
                quads.add(quad(v(1 - p, 1 - o, 1 - o), v(1 - p, 1 - o, o), v(1 - o, 1 - o, o), v(1 - o, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(1 - p, o, o), v(1 - p, o, 1 - o), v(1 - o, o, 1 - o), v(1 - o, o, o), spriteCable));
                quads.add(quad(v(1 - p, 1 - o, o), v(1 - p, o, o), v(1 - o, o, o), v(1 - o, 1 - o, o), spriteCable));
                quads.add(quad(v(1 - p, o, 1 - o), v(1 - p, 1 - o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, o, 1 - o), spriteCable));

                quads.add(quad(v(1 - p, 1 - q, 1 - q), v(1, 1 - q, 1 - q), v(1, 1 - q, q), v(1 - p, 1 - q, q), spriteSide));
                quads.add(quad(v(1 - p, q, q), v(1, q, q), v(1, q, 1 - q), v(1 - p, q, 1 - q), spriteSide));
                quads.add(quad(v(1 - p, 1 - q, q), v(1, 1 - q, q), v(1, q, q), v(1 - p, q, q), spriteSide));
                quads.add(quad(v(1 - p, q, 1 - q), v(1, q, 1 - q), v(1, 1 - q, 1 - q), v(1 - p, 1 - q, 1 - q), spriteSide));

                quads.add(quad(v(1 - p, q, 1 - q), v(1 - p, 1 - q, 1 - q), v(1 - p, 1 - q, q), v(1 - p, q, q), spriteConnector));
                quads.add(quad(v(1, q, 1 - q), v(1, q, q), v(1, 1 - q, q), v(1, 1 - q, 1 - q), spriteSide));
            } else {
                QuadSetting pattern = CablePatterns.findPattern(down, north, up, south);
                quads.add(quad(v(1 - o, o, o), v(1 - o, 1 - o, o), v(1 - o, 1 - o, 1 - o), v(1 - o, o, 1 - o), spriteGetter.apply(pattern.sprite()), pattern.rotation()));
            }

            if (west == CABLE) {
                quads.add(quad(v(o, 1 - o, 1 - o), v(o, 1 - o, o), v(0, 1 - o, o), v(0, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(o, o, o), v(o, o, 1 - o), v(0, o, 1 - o), v(0, o, o), spriteCable));
                quads.add(quad(v(o, 1 - o, o), v(o, o, o), v(0, o, o), v(0, 1 - o, o), spriteCable));
                quads.add(quad(v(o, o, 1 - o), v(o, 1 - o, 1 - o), v(0, 1 - o, 1 - o), v(0, o, 1 - o), spriteCable));
            } else if (west == BLOCK) {
                quads.add(quad(v(o, 1 - o, 1 - o), v(o, 1 - o, o), v(p, 1 - o, o), v(p, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(o, o, o), v(o, o, 1 - o), v(p, o, 1 - o), v(p, o, o), spriteCable));
                quads.add(quad(v(o, 1 - o, o), v(o, o, o), v(p, o, o), v(p, 1 - o, o), spriteCable));
                quads.add(quad(v(o, o, 1 - o), v(o, 1 - o, 1 - o), v(p, 1 - o, 1 - o), v(p, o, 1 - o), spriteCable));

                quads.add(quad(v(0, 1 - q, 1 - q), v(p, 1 - q, 1 - q), v(p, 1 - q, q), v(0, 1 - q, q), spriteSide));
                quads.add(quad(v(0, q, q), v(p, q, q), v(p, q, 1 - q), v(0, q, 1 - q), spriteSide));
                quads.add(quad(v(0, 1 - q, q), v(p, 1 - q, q), v(p, q, q), v(0, q, q), spriteSide));
                quads.add(quad(v(0, q, 1 - q), v(p, q, 1 - q), v(p, 1 - q, 1 - q), v(0, 1 - q, 1 - q), spriteSide));

                quads.add(quad(v(p, q, q), v(p, 1 - q, q), v(p, 1 - q, 1 - q), v(p, q, 1 - q), spriteConnector));
                quads.add(quad(v(0, q, q), v(0, q, 1 - q), v(0, 1 - q, 1 - q), v(0, 1 - q, q), spriteSide));
            } else {
                QuadSetting pattern = CablePatterns.findPattern(down, south, up, north);
                quads.add(quad(v(o, o, 1 - o), v(o, 1 - o, 1 - o), v(o, 1 - o, o), v(o, o, o), spriteGetter.apply(pattern.sprite()), pattern.rotation()));
            }

            if (north == CABLE) {
                quads.add(quad(v(o, 1 - o, o), v(1 - o, 1 - o, o), v(1 - o, 1 - o, 0), v(o, 1 - o, 0), spriteCable));
                quads.add(quad(v(o, o, 0), v(1 - o, o, 0), v(1 - o, o, o), v(o, o, o), spriteCable));
                quads.add(quad(v(1 - o, o, 0), v(1 - o, 1 - o, 0), v(1 - o, 1 - o, o), v(1 - o, o, o), spriteCable));
                quads.add(quad(v(o, o, o), v(o, 1 - o, o), v(o, 1 - o, 0), v(o, o, 0), spriteCable));
            } else if (north == BLOCK) {
                quads.add(quad(v(o, 1 - o, o), v(1 - o, 1 - o, o), v(1 - o, 1 - o, p), v(o, 1 - o, p), spriteCable));
                quads.add(quad(v(o, o, p), v(1 - o, o, p), v(1 - o, o, o), v(o, o, o), spriteCable));
                quads.add(quad(v(1 - o, o, p), v(1 - o, 1 - o, p), v(1 - o, 1 - o, o), v(1 - o, o, o), spriteCable));
                quads.add(quad(v(o, o, o), v(o, 1 - o, o), v(o, 1 - o, p), v(o, o, p), spriteCable));

                quads.add(quad(v(q, 1 - q, p), v(1 - q, 1 - q, p), v(1 - q, 1 - q, 0), v(q, 1 - q, 0), spriteSide));
                quads.add(quad(v(q, q, 0), v(1 - q, q, 0), v(1 - q, q, p), v(q, q, p), spriteSide));
                quads.add(quad(v(1 - q, q, 0), v(1 - q, 1 - q, 0), v(1 - q, 1 - q, p), v(1 - q, q, p), spriteSide));
                quads.add(quad(v(q, q, p), v(q, 1 - q, p), v(q, 1 - q, 0), v(q, q, 0), spriteSide));

                quads.add(quad(v(q, q, p), v(1 - q, q, p), v(1 - q, 1 - q, p), v(q, 1 - q, p), spriteConnector));
                quads.add(quad(v(q, q, 0), v(q, 1 - q, 0), v(1 - q, 1 - q, 0), v(1 - q, q, 0), spriteSide));
            } else {
                QuadSetting pattern = CablePatterns.findPattern(west, up, east, down);
                quads.add(quad(v(o, 1 - o, o), v(1 - o, 1 - o, o), v(1 - o, o, o), v(o, o, o), spriteGetter.apply(pattern.sprite()), pattern.rotation()));
            }

            if (south == CABLE) {
                quads.add(quad(v(o, 1 - o, 1), v(1 - o, 1 - o, 1), v(1 - o, 1 - o, 1 - o), v(o, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(o, o, 1 - o), v(1 - o, o, 1 - o), v(1 - o, o, 1), v(o, o, 1), spriteCable));
                quads.add(quad(v(1 - o, o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1 - o, 1), v(1 - o, o, 1), spriteCable));
                quads.add(quad(v(o, o, 1), v(o, 1 - o, 1), v(o, 1 - o, 1 - o), v(o, o, 1 - o), spriteCable));
            } else if (south == BLOCK) {
                quads.add(quad(v(o, 1 - o, 1 - p), v(1 - o, 1 - o, 1 - p), v(1 - o, 1 - o, 1 - o), v(o, 1 - o, 1 - o), spriteCable));
                quads.add(quad(v(o, o, 1 - o), v(1 - o, o, 1 - o), v(1 - o, o, 1 - p), v(o, o, 1 - p), spriteCable));
                quads.add(quad(v(1 - o, o, 1 - o), v(1 - o, 1 - o, 1 - o), v(1 - o, 1 - o, 1 - p), v(1 - o, o, 1 - p), spriteCable));
                quads.add(quad(v(o, o, 1 - p), v(o, 1 - o, 1 - p), v(o, 1 - o, 1 - o), v(o, o, 1 - o), spriteCable));

                quads.add(quad(v(q, 1 - q, 1), v(1 - q, 1 - q, 1), v(1 - q, 1 - q, 1 - p), v(q, 1 - q, 1 - p), spriteSide));
                quads.add(quad(v(q, q, 1 - p), v(1 - q, q, 1 - p), v(1 - q, q, 1), v(q, q, 1), spriteSide));
                quads.add(quad(v(1 - q, q, 1 - p), v(1 - q, 1 - q, 1 - p), v(1 - q, 1 - q, 1), v(1 - q, q, 1), spriteSide));
                quads.add(quad(v(q, q, 1), v(q, 1 - q, 1), v(q, 1 - q, 1 - p), v(q, q, 1 - p), spriteSide));

                quads.add(quad(v(q, 1 - q, 1 - p), v(1 - q, 1 - q, 1 - p), v(1 - q, q, 1 - p), v(q, q, 1 - p), spriteConnector));
                quads.add(quad(v(q, 1 - q, 1), v(q, q, 1), v(1 - q, q, 1), v(1 - q, 1 - q, 1), spriteSide));
            } else {
                QuadSetting pattern = CablePatterns.findPattern(west, down, east, up);
                quads.add(quad(v(o, o, 1 - o), v(1 - o, o, 1 - o), v(1 - o, 1 - o, 1 - o), v(o, 1 - o, 1 - o), spriteGetter.apply(pattern.sprite()), pattern.rotation()));
            }
        }

        // Render the facade if we have one in addition to the cable above. Note that the facade comes from the model data property
        // (FACADEID)
        BlockState facadeId = extraData.get(CableBlock.FACADEID);
        if (facadeId != null) {
            BakedModel model = Minecraft.getInstance().getBlockRenderer().getBlockModelShaper().getBlockModel(facadeId);
            ChunkRenderTypeSet renderTypes = model.getRenderTypes(facadeId, rand, extraData);
            if (layer == null || renderTypes.contains(layer)) { // always render in the null layer or the block-breaking textures don't show up
                try {
                    quads.addAll(model.getQuads(state, side, rand, ModelData.EMPTY, layer));
                } catch (Exception ignored) {
                }
            }
        }

        return quads;
    }

    @Override
    public boolean useAmbientOcclusion() {
        return true;
    }

    @Override
    public boolean isGui3d() {
        return false;
    }

    @Override
    public boolean isCustomRenderer() {
        return false;
    }

    // Because we can potentially mimic other blocks we need to render on all render types
    @Override
    @Nonnull
    public ChunkRenderTypeSet getRenderTypes(@NotNull BlockState state, @NotNull RandomSource rand, @NotNull ModelData data) {
        return ChunkRenderTypeSet.all();
    }

    @Nonnull
    @Override
    public TextureAtlasSprite getParticleIcon() {
        return spriteNormalCable == null
                ? Minecraft.getInstance().getTextureAtlas(InventoryMenu.BLOCK_ATLAS).apply((new ResourceLocation("minecraft", "missingno")))
                : spriteNormalCable;
    }

    // To let our cable/facade render correctly as an item (both in inventory and on the ground) we
    // get the correct transforms from the context
    @Nonnull
    @Override
    public ItemTransforms getTransforms() {
        return context.getTransforms();
    }

    @Nonnull
    @Override
    public ItemOverrides getOverrides() {
        return ItemOverrides.EMPTY;
    }

}
```

#### The CablePatterns helper

```java

public class CablePatterns {

    // This map takes a pattern of four directions (excluding the one we are looking at) and returns the sprite index
    // and rotation for the quad that we are looking at.
    static final Map<Pattern, QuadSetting> PATTERNS = new HashMap<>();

    // Given a pattern of four directions (excluding the one we are looking at) we return the sprite index and rotation
    // for the quad that we are looking at.
    public static QuadSetting findPattern(ConnectorType s1, ConnectorType s2, ConnectorType s3, ConnectorType s4) {
        return PATTERNS.get(new Pattern(s1 != NONE, s2 != NONE, s3 != NONE, s4 != NONE));
    }

    // This enum represents the type of sprite (texture)
    public enum SpriteIdx {
        SPRITE_NONE,
        SPRITE_END,
        SPRITE_STRAIGHT,
        SPRITE_CORNER,
        SPRITE_THREE,
        SPRITE_CROSS
    }

    // This enum represents the type of sprite (texture) as well as the rotation for that sprite
    public record QuadSetting(SpriteIdx sprite, int rotation) {

        public static QuadSetting of(SpriteIdx sprite, int rotation) {
            return new QuadSetting(sprite, rotation);
        }
    }

    // A pattern represents a configuration (cable or no cable) for the four directions excluding the one we are looking at
    public record Pattern(boolean s1, boolean s2, boolean s3, boolean s4) {

        public static Pattern of(boolean s1, boolean s2, boolean s3, boolean s4) {
            return new Pattern(s1, s2, s3, s4);
        }
    }
}
```

#### The BakedModelHelper

BakedModelHelper是一个辅助类，有一些创建四边形的辅助方法，我们使用这个类来创建线缆的四边形模型。

```java


public class BakedModelHelper {

    public static BakedQuad quad(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, TextureAtlasSprite sprite, int rotation) {
        return switch (rotation) {
            case 0 -> quad(v1, v2, v3, v4, sprite);
            case 1 -> quad(v2, v3, v4, v1, sprite);
            case 2 -> quad(v3, v4, v1, v2, sprite);
            case 3 -> quad(v4, v1, v2, v3, sprite);
            default -> quad(v1, v2, v3, v4, sprite);
        };
    }

    public static BakedQuad quad(Vec3 v1, Vec3 v2, Vec3 v3, Vec3 v4, TextureAtlasSprite sprite) {
        Vec3 normal = v3.subtract(v2).cross(v1.subtract(v2)).normalize();

        BakedQuad[] quad = new BakedQuad[1];
        QuadBakingVertexConsumer builder = new QuadBakingVertexConsumer(q -> quad[0] = q);
        builder.setSprite(sprite);
        builder.setDirection(Direction.getNearest(normal.x, normal.y, normal.z));
        putVertex(builder, normal, v1.x, v1.y, v1.z, 0, 0, sprite);
        putVertex(builder, normal, v2.x, v2.y, v2.z, 0, 16, sprite);
        putVertex(builder, normal, v3.x, v3.y, v3.z, 16, 16, sprite);
        putVertex(builder, normal, v4.x, v4.y, v4.z, 16, 0, sprite);
        return quad[0];
    }

    private static void putVertex(VertexConsumer builder, Position normal,
                                 double x, double y, double z, float u, float v,
                                 TextureAtlasSprite sprite) {
        float iu = sprite.getU(u);
        float iv = sprite.getV(v);
        builder.vertex(x, y, z)
                .uv(iu, iv)
                .uv2(0, 0)
                .color(1.0f, 1.0f, 1.0f, 1.0f)
                .normal((float) normal.x(), (float) normal.y(), (float) normal.z())
                .endVertex();
    }

    public static Vec3 v(double x, double y, double z) {
        return new Vec3(x, y, z);
    }
}
```



### 数据生成Data Generation

最后一件我们需要解释的事情是Data Generation。我们不会在这里进行详细的介绍，因为你应该知道它是如何工作的。你可以在github查看详细的代码。然而，然而我想解释一下关于我们使用的烘焙模型系统的模型如何进行数据生成。

为了生成电缆和facade方块的json，我们可以在`TutBlockStates`中使用以下的代码，因为我们需要一个为我们模型自定义的builder，所以创建了一个builder的类，这个类叫做CableLoaderBuilder继承自CustomLoaderBuilder，CableLoaderBuilder使用ResourceLocation 作为加载id，BlockModelBuilder是其父类，ExistingFileHelper和一个布尔值指出我们是否生成一个facade方块，CableLoaderBuilder需要重写toJson的方法增加facade的属性，facade属性用于CableModelLoader中决定我们是否生成一个电缆或者facade方块。

在registerCable（）和registerFacade（）中，我们创建了一个BlockModelBuilder，父类为cube。然后，我们将自定义加载器设置为我们的CableLoaderBuilder，并设置facade属性，最后我们在BlockModelBuilder和block中调用simpleBlock 。

因为我们使用了原版的cube作为父模型，所以我们将会继承该项目的正确转化，这意味着电缆方块和facade方块将会在背包和地面上得到正确的渲染（也是因为我们在烘焙模型中使用content来获取变换）

```java

public class TutBlockStates extends BlockStateProvider {

    ...

    @Override
    protected void registerStatesAndModels() {
        ...
        registerCable();
        registerFacade();
    }

    private void registerCable() {
        BlockModelBuilder model = models().getBuilder("cable")
                .parent(models().getExistingFile(mcLoc("cube")))
                .customLoader((builder, helper) -> new CableLoaderBuilder(CableModelLoader.GENERATOR_LOADER, builder, helper, false))
                .end();
        simpleBlock(Registration.CABLE_BLOCK.get(), model);
    }

    private void registerFacade() {
        BlockModelBuilder model = models().getBuilder("facade")
                .parent(models().getExistingFile(mcLoc("cube")))
                .customLoader((builder, helper) -> new CableLoaderBuilder(CableModelLoader.GENERATOR_LOADER, builder, helper, true))
                .end();
        simpleBlock(Registration.FACADE_BLOCK.get(), model);
    }

    ...
    
    public static class CableLoaderBuilder extends CustomLoaderBuilder<BlockModelBuilder> {

        private final boolean facade;

        public CableLoaderBuilder(ResourceLocation loader, BlockModelBuilder parent, ExistingFileHelper existingFileHelper,
                                  boolean facade) {
            super(loader, parent, existingFileHelper);
            this.facade = facade;
        }

        @Override
        public JsonObject toJson(JsonObject json) {
            JsonObject obj = super.toJson(json);
            obj.addProperty("facade", facade);
            return obj;
        }
    }
}
```

