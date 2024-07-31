CompilationUnit(  
    imports=[],         # 导入语句列表，此处为空，表示没有导入任何包或类  
    package=None,       # 包声明，此处为None，表示没有包声明  
    types=[             # 类型列表，通常包含类、接口、枚举等的定义  
        ClassDeclaration(  # 类的声明  
            annotations=[],  # 注解列表，此处为空  
            body=[           # 类的主体，包含方法、字段等成员  
                MethodDeclaration(  # 方法的声明  
                    annotations=[],  # 注解列表，此处为空  
                    body=[       # 方法的主体  
                        LocalVariableDeclaration(  # 局部变量声明  
                            annotations=[],  # 注解列表，此处为空  
                            declarators=[  # 变量声明符列表  
                                VariableDeclarator(  # 变量声明符  
                                    dimensions=[],  # 数组维度，此处为空，表示不是数组  
                                    initializer=BinaryOperation(  # 初始化表达式，一个二元操作  
                                        operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]),  # 左操作数，引用变量a  
                                        operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]),  # 右操作数，引用变量b  
                                        operator=+  # 操作符，加号  
                                    ),  
                                    name=sum  # 变量名  
                                )  
                            ],  
                            modifiers=set(),  # 修饰符列表，此处为空  
                            type=BasicType(dimensions=[], name=int)  # 变量类型，基本类型int  
                        ),  
                        ReturnStatement(  # 返回语句  
                            expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]),  # 返回的表达式，引用变量sum  
                            label=None  # 语句标签，此处为None  
                        )  
                    ],  
                    documentation=None,  # 文档字符串，此处为None  
                    modifiers={'public'},  # 修饰符列表，包含'public'  
                    name=add,  # 方法名  
                    parameters=[  # 参数列表  
                        FormalParameter(  # 形式参数  
                            annotations=[],  # 注解列表，此处为空  
                            modifiers=set(),  # 修饰符列表，此处为空  
                            name=a,  # 参数名  
                            type=BasicType(dimensions=[], name=int),  # 参数类型，基本类型int  
                            varargs=False  # 是否为可变参数，此处为False  
                        ),  
                        FormalParameter(  # 形式参数  
                            annotations=[],  # 注解列表，此处为空  
                            modifiers=set(),  # 修饰符列表，此处为空  
                            name=b,  # 参数名  
                            type=BasicType(dimensions=[], name=int),  # 参数类型，基本类型int  
                            varargs=False  # 是否为可变参数，此处为False  
                        )  
                    ],  
                    return_type=BasicType(dimensions=[], name=int),  # 返回类型，基本类型int  
                    throws=None,  # 异常列表，此处为None  
                    type_parameters=None  # 类型参数列表，此处为None  
                )  
            ],  
            documentation=None,  # 文档字符串，此处为None  
            extends=None,  # 父类，此处为None，表示没有继承其他类  
            implements=None,  # 实现的接口列表，此处为None  
            modifiers={'public'},  # 修饰符列表，包含'public'  
            name=Example,  # 类名  
            type_parameters=None  # 类型参数列表，此处为None  
        )  
    ]  
)



CompilationUnit(
    imports=[],
    package=None, 
    types=[
        ClassDeclaration(
            annotations=[], 
            body=[
               MethodDeclaration(
                    annotations=[], 
                    body=[
                        LocalVariableDeclaration(
                                annotations=[], 
                                declarators=[ # 变量声明符列表  
                                    VariableDeclarator(
                                        dimensions=[], 
                                        initializer=Literal(postfix_operators=[], prefix_operators=[], qualifier=None, selectors=[], value=0), 
                                        name=sum)
                                    ], 
                                modifiers=set(), # 修饰符列表，此处为空  
                                type=BasicType(dimensions=[], name=int)
                            ), 
                        StatementExpression(
                            expression=Assignment(
                                expressionl=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]),
                                type==, 
                                value=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), 
                                operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+)
                                ), 
                            label=None), 
                            ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)
                        ], 
                        documentation=None, 
                        modifiers={'public'}, 
                        name=add, 
                        parameters=[
                            FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False),
                            FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)
                            ], 
                        return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None), 

                MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=delete, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None)
            ], 
            documentation=None, extends=None, implements=None, modifiers={'public'}, name=Example, type_parameters=None)])




    CompilationUnit(imports=[], package=None, types=[ClassDeclaration(annotations=[], body=[MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=add, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None), MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=delete, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None)], documentation=None, extends=None, implements=None, modifiers={'public'}, name=Example, type_parameters=None)])
    None
    ClassDeclaration(annotations=[], body=[MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=add, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None), MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=delete, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None)], documentation=None, extends=None, implements=None, modifiers={'public'}, name=Example, type_parameters=None)      
        {'public'}
        None
        Example
        MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=add, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None)
            None
            {'public'}
            None
            BasicType(dimensions=[], name=int)
                int
            add
            FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False)
                set()
                BasicType(dimensions=[], name=int)
                    int
                a
                False
            FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)
                set()
                BasicType(dimensions=[], name=int)
                    int
                b
                False
            None
            LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int))
                set()
                BasicType(dimensions=[], name=int)
                    int
                VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+), name=sum)
                    sum
                    BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=+)
                        +
                        MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[])

                            a
                        MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[])

                            b
            ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)
                None
                MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[])

                    sum
        MethodDeclaration(annotations=[], body=[LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int)), ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)], documentation=None, modifiers={'public'}, name=delete, parameters=[FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False), FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)], return_type=BasicType(dimensions=[], name=int), throws=None, type_parameters=None)
            None
            {'public'}
            None
            BasicType(dimensions=[], name=int)
                int
            delete
            FormalParameter(annotations=[], modifiers=set(), name=a, type=BasicType(dimensions=[], name=int), varargs=False)
                set()
                BasicType(dimensions=[], name=int)
                    int
                a
                False
            FormalParameter(annotations=[], modifiers=set(), name=b, type=BasicType(dimensions=[], name=int), varargs=False)
                set()
                BasicType(dimensions=[], name=int)
                    int
                b
                False
            None
            LocalVariableDeclaration(annotations=[], declarators=[VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-), name=sum)], modifiers=set(), type=BasicType(dimensions=[], name=int))
                set()
                BasicType(dimensions=[], name=int)
                    int
                VariableDeclarator(dimensions=[], initializer=BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-), name=sum)
                    sum
                    BinaryOperation(operandl=MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operandr=MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), operator=-)
                        -
                        MemberReference(member=a, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[])

                            a
                        MemberReference(member=b, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[])

                            b
            ReturnStatement(expression=MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[]), label=None)
                None
                MemberReference(member=sum, postfix_operators=[], prefix_operators=[], qualifier=, selectors=[])

                    sum
        None
        None
        None